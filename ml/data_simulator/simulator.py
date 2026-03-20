"""
AnomaNet Data Simulator — ml/data_simulator/simulator.py

Generates 100,000 synthetic bank transactions:
  - 95,000 clean transactions (realistic Indian banking behaviour)
  - 5,000 fraud transactions (1,000 per typology × 5 typologies)

Output:
  - data/transactions.parquet        — full transaction ledger
  - data/accounts.parquet            — account master
  - data/customers.parquet           — customer KYC master
  - data/labels.parquet              — fraud ground-truth labels
  - data/neo4j_edges.parquet         — edge list ready for Neo4j import
  - data/neo4j_nodes.parquet         — node list ready for Neo4j import

Usage:
  python -m data_simulator.simulator               # generates all 100k
  python -m data_simulator.simulator --count 10000 # quick dev run
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from faker import Faker

# ── Import the five scenario generators ─────────────────────────────────────
from data_simulator.scenarios.layering import generate_layering_cluster
from data_simulator.scenarios.circular import generate_circular_cluster
from data_simulator.scenarios.structuring import generate_structuring_cluster
from data_simulator.scenarios.dormant_activation import generate_dormant_cluster
from data_simulator.scenarios.profile_mismatch_gen import generate_profile_mismatch_cluster

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

fake = Faker("en_IN")
Faker.seed(42)
random.seed(42)
np.random.seed(42)

# ── Constants ────────────────────────────────────────────────────────────────

TOTAL_TRANSACTIONS = 100_000
FRAUD_PER_TYPOLOGY = 1_000
FRAUD_TOTAL = FRAUD_PER_TYPOLOGY * 5          # 5,000
CLEAN_TOTAL = TOTAL_TRANSACTIONS - FRAUD_TOTAL  # 95,000

# Simulation window: last 90 days
SIM_END   = datetime.now(tz=timezone.utc).replace(microsecond=0)
SIM_START = SIM_END - timedelta(days=90)

# Indian banking channels with realistic weights
CHANNELS = ["UPI", "NEFT", "IMPS", "RTGS", "CASH", "BRANCH", "SWIFT"]
CHANNEL_WEIGHTS = [0.42, 0.22, 0.18, 0.08, 0.05, 0.03, 0.02]

# IFSC prefixes for 12 major Indian banks / branches
IFSC_PREFIXES = [
    "HDFC0001", "HDFC0002", "HDFC0003",
    "ICIC0001", "ICIC0002",
    "SBIN0001", "SBIN0002", "SBIN0003",
    "AXIS0001", "AXIS0002",
    "KKBK0001", "PUNB0001",
]

ACCOUNT_TYPES = ["SAVINGS", "CURRENT", "OD", "LOAN", "NRE", "NRO"]
ACCOUNT_TYPE_WEIGHTS = [0.55, 0.25, 0.08, 0.05, 0.04, 0.03]

KYC_TIERS = ["LOW", "MEDIUM", "HIGH", "PEP"]
KYC_TIER_WEIGHTS = [0.50, 0.35, 0.13, 0.02]

OCCUPATIONS = [
    "Salaried Employee", "Business Owner", "Self Employed Professional",
    "Government Employee", "Retired", "Student", "Agriculturist",
    "Trader", "Freelancer", "Homemaker",
]

# Declared monthly income ranges by KYC tier (INR)
INCOME_RANGES = {
    "LOW":    (8_000,   40_000),
    "MEDIUM": (40_000,  2_00_000),
    "HIGH":   (2_00_000, 20_00_000),
    "PEP":    (5_00_000, 50_00_000),
}

# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class Customer:
    id: str
    name: str
    kyc_id: str
    risk_tier: str
    city: str
    state: str
    occupation: str
    segment: str
    declared_monthly_income: float


@dataclass
class Account:
    id: str
    customer_id: str
    account_type: str
    kyc_risk_tier: str
    declared_monthly_income: float
    declared_occupation: str
    open_date: str
    last_transaction_date: Optional[str]
    is_dormant: bool
    dormant_since: Optional[str]
    status: str
    branch_id: str


@dataclass
class Transaction:
    id: str
    reference_number: str
    source_account_id: str
    dest_account_id: str
    amount: float
    channel: str
    initiated_at: datetime
    settled_at: Optional[datetime]
    branch_id: str
    status: str
    metadata: dict
    # fraud labels
    is_fraud: bool = False
    fraud_type: Optional[str] = None
    fraud_cluster_id: Optional[str] = None


# ── Utility helpers ──────────────────────────────────────────────────────────

def _new_uuid() -> str:
    return str(uuid.uuid4())


def _new_account_number() -> str:
    """12-digit account number."""
    return str(random.randint(100_000_000_000, 999_999_999_999))


def _random_ifsc() -> str:
    prefix = random.choice(IFSC_PREFIXES)
    suffix = f"{random.randint(0, 9999):04d}"
    return f"{prefix}{suffix}"


def _random_timestamp(start: datetime, end: datetime) -> datetime:
    delta = end - start
    return start + timedelta(seconds=random.randint(0, int(delta.total_seconds())))


def _settlement_delay(channel: str, initiated_at: datetime) -> Optional[datetime]:
    """Realistic settlement delay per channel."""
    delays = {
        "UPI":    timedelta(seconds=random.randint(1, 30)),
        "IMPS":   timedelta(seconds=random.randint(5, 120)),
        "NEFT":   timedelta(hours=random.choice([0, 2, 4, 6])),
        "RTGS":   timedelta(minutes=random.randint(15, 90)),
        "CASH":   None,   # cash is immediate, no settle record
        "BRANCH": timedelta(hours=random.randint(1, 4)),
        "SWIFT":  timedelta(days=random.randint(1, 3)),
    }
    delay = delays.get(channel)
    return (initiated_at + delay) if delay else None


def _channel_for_account_type(account_type: str) -> str:
    """Bias channel selection by account type."""
    if account_type in ("NRE", "NRO"):
        return random.choices(["SWIFT", "NEFT", "RTGS"], weights=[0.5, 0.3, 0.2])[0]
    if account_type == "CURRENT":
        return random.choices(CHANNELS, weights=[0.2, 0.3, 0.15, 0.25, 0.05, 0.03, 0.02])[0]
    return random.choices(CHANNELS, weights=CHANNEL_WEIGHTS)[0]


def _realistic_amount(kyc_tier: str) -> float:
    """Draw a transaction amount realistic for the KYC tier."""
    low, high = INCOME_RANGES[kyc_tier]
    # Most transactions are small; a few are large (power-law-ish)
    if random.random() < 0.70:
        amount = random.uniform(low * 0.01, low * 0.5)
    elif random.random() < 0.90:
        amount = random.uniform(low * 0.5, high * 0.3)
    else:
        amount = random.uniform(high * 0.3, high * 1.2)
    return round(amount, 2)


def _utr() -> str:
    return f"UTR{random.randint(100_000_000_000_000, 999_999_999_999_999)}"


def _swift_ref() -> str:
    return f"SWIFT{random.randint(10_000_000, 99_999_999)}"


def _build_metadata(channel: str, tx_id: str) -> dict:
    meta: dict = {}
    if channel in ("NEFT", "RTGS", "IMPS"):
        meta["utr"] = _utr()
    if channel == "SWIFT":
        meta["swift_ref"] = _swift_ref()
    if channel == "UPI":
        meta["upi_txn_id"] = f"UPI{random.randint(10**15, 10**16 - 1)}"
        meta["device_id"] = fake.uuid4()
    return meta


# ── Customer and Account factory ─────────────────────────────────────────────

def _make_customer(kyc_tier: Optional[str] = None) -> Customer:
    tier = kyc_tier or random.choices(KYC_TIERS, weights=KYC_TIER_WEIGHTS)[0]
    income_lo, income_hi = INCOME_RANGES[tier]
    city  = fake.city()
    state = fake.state()
    return Customer(
        id=_new_uuid(),
        name=fake.name(),
        kyc_id=f"PAN{''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=5))}{random.randint(1000, 9999)}{''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=1))}",
        risk_tier=tier,
        city=city,
        state=state,
        occupation=random.choice(OCCUPATIONS),
        segment="RETAIL" if tier in ("LOW", "MEDIUM") else "WEALTH",
        declared_monthly_income=round(random.uniform(income_lo, income_hi), 2),
    )


def _make_account(customer: Customer, open_days_ago: int = None, force_dormant: bool = False) -> Account:
    if open_days_ago is None:
        open_days_ago = random.randint(180, 3650)
    open_date_dt  = SIM_END - timedelta(days=open_days_ago)

    is_dormant = False
    dormant_since = None
    last_tx_date  = None

    if force_dormant:
        # dormant for 14-24 months
        dormant_days = random.randint(420, 730)
        dormant_since_dt = SIM_END - timedelta(days=dormant_days)
        dormant_since = dormant_since_dt.strftime("%Y-%m-%d")
        last_tx_date  = dormant_since
        is_dormant    = True
    elif random.random() < 0.06:
        # ~6% of accounts are naturally dormant
        dormant_days = random.randint(365, 1000)
        dormant_since_dt = SIM_END - timedelta(days=dormant_days)
        dormant_since = dormant_since_dt.strftime("%Y-%m-%d")
        last_tx_date  = dormant_since
        is_dormant    = True
    else:
        last_tx_date = (SIM_END - timedelta(days=random.randint(0, 30))).strftime("%Y-%m-%d")

    return Account(
        id=_new_account_number(),
        customer_id=customer.id,
        account_type=random.choices(ACCOUNT_TYPES, weights=ACCOUNT_TYPE_WEIGHTS)[0],
        kyc_risk_tier=customer.risk_tier,
        declared_monthly_income=customer.declared_monthly_income,
        declared_occupation=customer.occupation,
        open_date=open_date_dt.strftime("%Y-%m-%d"),
        last_transaction_date=last_tx_date,
        is_dormant=is_dormant,
        dormant_since=dormant_since,
        status="DORMANT" if is_dormant else "ACTIVE",
        branch_id=_random_ifsc(),
    )


# ── Clean transaction factory ─────────────────────────────────────────────────

def _make_clean_transaction(
    accounts: list[Account],
    account_index: dict[str, Account],
) -> Transaction:
    src = random.choice(accounts)
    # Pick a counterparty — 70% existing accounts, 30% any account
    if random.random() < 0.70:
        dst = random.choice(accounts)
    else:
        dst = random.choice(accounts)

    while dst.id == src.id or dst.is_dormant:
        dst = random.choice(accounts)

    channel    = _channel_for_account_type(src.account_type)
    initiated  = _random_timestamp(SIM_START, SIM_END)
    settled    = _settlement_delay(channel, initiated)
    amount     = _realistic_amount(src.kyc_risk_tier)
    tx_id      = _new_uuid()

    return Transaction(
        id=tx_id,
        reference_number=f"{channel}{initiated.strftime('%Y%m%d')}{random.randint(100000, 999999)}",
        source_account_id=src.id,
        dest_account_id=dst.id,
        amount=amount,
        channel=channel,
        initiated_at=initiated,
        settled_at=settled,
        branch_id=src.branch_id,
        status="SETTLED" if settled else "SETTLED",
        metadata=_build_metadata(channel, tx_id),
        is_fraud=False,
        fraud_type=None,
        fraud_cluster_id=None,
    )


# ── Main generation orchestrator ─────────────────────────────────────────────

def generate_universe(n_accounts: int = 2_000) -> tuple[
    list[Customer], list[Account], list[Transaction]
]:
    """
    Build the full account + customer universe, then generate all transactions.
    Returns (customers, accounts, transactions).
    """
    log.info("Building customer and account universe (%d accounts)...", n_accounts)

    customers: list[Customer] = []
    accounts:  list[Account]  = []

    for _ in range(n_accounts):
        cust = _make_customer()
        acct = _make_account(cust)
        customers.append(cust)
        accounts.append(acct)

    account_index = {a.id: a for a in accounts}
    active_accounts = [a for a in accounts if not a.is_dormant]

    log.info(
        "Universe: %d customers | %d accounts | %d active | %d dormant",
        len(customers), len(accounts),
        len(active_accounts),
        len(accounts) - len(active_accounts),
    )

    all_transactions: list[Transaction] = []

    # ── 1. Inject fraud scenarios ────────────────────────────────────────────
    log.info("Injecting fraud scenarios (%d × 5 typologies)...", FRAUD_PER_TYPOLOGY)

    shared_pool = active_accounts[:]

    txns_layering, accts_l, custs_l = generate_layering_cluster(
        n_clusters=FRAUD_PER_TYPOLOGY // 5,   # each cluster ≈ 5 txns → 200 clusters
        shared_pool=shared_pool,
        sim_end=SIM_END,
    )
    txns_circular, accts_c, custs_c = generate_circular_cluster(
        n_clusters=FRAUD_PER_TYPOLOGY // 4,
        shared_pool=shared_pool,
        sim_end=SIM_END,
    )
    txns_structuring, accts_s, custs_s = generate_structuring_cluster(
        n_clusters=FRAUD_PER_TYPOLOGY // 3,
        shared_pool=shared_pool,
        sim_end=SIM_END,
    )
    txns_dormant, accts_d, custs_d = generate_dormant_cluster(
        n_clusters=FRAUD_PER_TYPOLOGY // 4,
        sim_end=SIM_END,
    )
    txns_profile, accts_p, custs_p = generate_profile_mismatch_cluster(
        n_clusters=FRAUD_PER_TYPOLOGY // 5,
        sim_end=SIM_END,
    )

    # Collect all generated fraud accounts into universe
    for acct_list in [accts_l, accts_c, accts_s, accts_d, accts_p]:
        accounts.extend(acct_list)
        active_accounts.extend([a for a in acct_list if not a.is_dormant])
    for cust_list in [custs_l, custs_c, custs_s, custs_d, custs_p]:
        customers.extend(cust_list)

    fraud_txns = (
        txns_layering + txns_circular +
        txns_structuring + txns_dormant + txns_profile
    )
    actual_fraud = len(fraud_txns)
    log.info("Fraud transactions generated: %d", actual_fraud)

    # ── 2. Generate clean transactions ───────────────────────────────────────
    clean_count = TOTAL_TRANSACTIONS - actual_fraud
    log.info("Generating %d clean transactions...", clean_count)

    all_accounts_snapshot = [a for a in accounts if not a.is_dormant]

    for i in range(clean_count):
        if i % 10_000 == 0 and i > 0:
            log.info("  clean transactions: %d / %d", i, clean_count)
        all_transactions.append(
            _make_clean_transaction(all_accounts_snapshot, account_index)
        )

    all_transactions.extend(fraud_txns)
    random.shuffle(all_transactions)  # mix fraud into the clean stream

    log.info("Total transactions: %d (fraud: %d, clean: %d)",
             len(all_transactions), actual_fraud, len(all_transactions) - actual_fraud)

    return customers, accounts, all_transactions


# ── Serialisation to parquet ──────────────────────────────────────────────────

def _transactions_to_df(transactions: list[Transaction]) -> pd.DataFrame:
    rows = []
    for t in transactions:
        rows.append({
            "id":                  t.id,
            "reference_number":    t.reference_number,
            "source_account_id":   t.source_account_id,
            "dest_account_id":     t.dest_account_id,
            "amount":              t.amount,
            "channel":             t.channel,
            "initiated_at":        t.initiated_at.isoformat(),
            "settled_at":          t.settled_at.isoformat() if t.settled_at else None,
            "branch_id":           t.branch_id,
            "status":              t.status,
            "metadata":            str(t.metadata),
            "is_fraud":            t.is_fraud,
            "fraud_type":          t.fraud_type,
            "fraud_cluster_id":    t.fraud_cluster_id,
        })
    return pd.DataFrame(rows)


def _accounts_to_df(accounts: list[Account]) -> pd.DataFrame:
    rows = []
    for a in accounts:
        rows.append({
            "id":                       a.id,
            "customer_id":              a.customer_id,
            "account_type":             a.account_type,
            "kyc_risk_tier":            a.kyc_risk_tier,
            "declared_monthly_income":  a.declared_monthly_income,
            "declared_occupation":      a.declared_occupation,
            "open_date":                a.open_date,
            "last_transaction_date":    a.last_transaction_date,
            "is_dormant":               a.is_dormant,
            "dormant_since":            a.dormant_since,
            "status":                   a.status,
            "branch_id":                a.branch_id,
        })
    return pd.DataFrame(rows)


def _customers_to_df(customers: list[Customer]) -> pd.DataFrame:
    rows = []
    for c in customers:
        rows.append({
            "id":                       c.id,
            "name":                     c.name,
            "kyc_id":                   c.kyc_id,
            "risk_tier":                c.risk_tier,
            "city":                     c.city,
            "state":                    c.state,
            "occupation":               c.occupation,
            "segment":                  c.segment,
            "declared_monthly_income":  c.declared_monthly_income,
        })
    return pd.DataFrame(rows)


def _build_neo4j_export(
    transactions: list[Transaction],
    accounts: list[Account],
    customers: list[Customer],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build Neo4j-ready node and edge lists.
    Nodes: Account, Customer, Branch
    Edges: TRANSFERRED_TO, OWNS, BELONGS_TO
    """
    # Account nodes
    acct_nodes = []
    branch_ids_seen = set()

    for a in accounts:
        acct_nodes.append({
            "node_type":  "Account",
            "id":         a.id,
            "account_type": a.account_type,
            "branch_id":  a.branch_id,
            "kyc_risk_tier": a.kyc_risk_tier,
            "is_dormant": a.is_dormant,
            "status":     a.status,
            "anoma_score": 0.0,   # initialised to 0; updated at runtime
        })
        branch_ids_seen.add(a.branch_id)

    # Customer nodes
    cust_nodes = [
        {
            "node_type":  "Customer",
            "id":         c.id,
            "name":       c.name,
            "kyc_id":     c.kyc_id,
            "risk_tier":  c.risk_tier,
            "city":       c.city,
            "occupation": c.occupation,
            "segment":    c.segment,
        }
        for c in customers
    ]

    # Branch nodes
    branch_nodes = [
        {"node_type": "Branch", "id": bid, "ifsc": bid}
        for bid in branch_ids_seen
    ]

    all_nodes = pd.DataFrame(acct_nodes + cust_nodes + branch_nodes)

    # Edges
    edges = []
    for t in transactions:
        edges.append({
            "edge_type":  "TRANSFERRED_TO",
            "source":     t.source_account_id,
            "target":     t.dest_account_id,
            "tx_id":      t.id,
            "amount":     t.amount,
            "timestamp":  t.initiated_at.isoformat(),
            "channel":    t.channel,
            "branch_id":  t.branch_id,
            "is_fraud":   t.is_fraud,
            "fraud_type": t.fraud_type,
        })

    acct_lookup = {a.id: a for a in accounts}
    for a in accounts:
        edges.append({
            "edge_type":  "OWNS",
            "source":     a.customer_id,
            "target":     a.id,
            "tx_id":      None,
            "amount":     None,
            "timestamp":  None,
            "channel":    None,
            "branch_id":  None,
            "is_fraud":   False,
            "fraud_type": None,
        })
        edges.append({
            "edge_type":  "BELONGS_TO",
            "source":     a.id,
            "target":     a.branch_id,
            "tx_id":      None,
            "amount":     None,
            "timestamp":  None,
            "channel":    None,
            "branch_id":  None,
            "is_fraud":   False,
            "fraud_type": None,
        })

    all_edges = pd.DataFrame(edges)
    return all_nodes, all_edges


# ── Entry point ───────────────────────────────────────────────────────────────

def run(output_dir: str = "data", total_override: Optional[int] = None):
    global TOTAL_TRANSACTIONS, CLEAN_TOTAL
    if total_override:
        TOTAL_TRANSACTIONS = total_override
        CLEAN_TOTAL = TOTAL_TRANSACTIONS - FRAUD_TOTAL

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    customers, accounts, transactions = generate_universe(n_accounts=2_000)

    log.info("Serialising to parquet...")

    tx_df   = _transactions_to_df(transactions)
    acct_df = _accounts_to_df(accounts)
    cust_df = _customers_to_df(customers)

    # Labels (separate file for clean ML training splits)
    labels_df = tx_df[["id", "is_fraud", "fraud_type", "fraud_cluster_id"]].copy()

    nodes_df, edges_df = _build_neo4j_export(transactions, accounts, customers)

    tx_df.to_parquet(out / "transactions.parquet", index=False)
    acct_df.to_parquet(out / "accounts.parquet", index=False)
    cust_df.to_parquet(out / "customers.parquet", index=False)
    labels_df.to_parquet(out / "labels.parquet", index=False)
    nodes_df.to_parquet(out / "neo4j_nodes.parquet", index=False)
    edges_df.to_parquet(out / "neo4j_edges.parquet", index=False)

    # Print quick summary
    fraud_counts = tx_df[tx_df["is_fraud"]]["fraud_type"].value_counts()
    log.info("=" * 60)
    log.info("SIMULATION COMPLETE")
    log.info("  Transactions : %d", len(tx_df))
    log.info("  Accounts     : %d", len(acct_df))
    log.info("  Customers    : %d", len(cust_df))
    log.info("  Fraud by type:")
    for ftype, cnt in fraud_counts.items():
        log.info("    %-25s %d", ftype, cnt)
    log.info("  Output dir   : %s", out.resolve())
    log.info("=" * 60)

    return tx_df, acct_df, cust_df, labels_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AnomaNet data simulator")
    parser.add_argument("--count", type=int, default=None,
                        help="Override total transaction count (default: 100,000)")
    parser.add_argument("--output", type=str, default="data",
                        help="Output directory for parquet files")
    args = parser.parse_args()
    run(output_dir=args.output, total_override=args.count)