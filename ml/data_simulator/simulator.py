"""
AnomaNet Data Simulator — ml/data_simulator/simulator.py

All shared types and helpers live in models.py.
This file only orchestrates generation and serialisation.

Usage:
  python -m data_simulator.simulator               # full 100k
  python -m data_simulator.simulator --count 10000 # quick dev run
  python -m data_simulator.simulator --output data  # custom output dir
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from data_simulator.models import (
    Account, Customer, Transaction,
    SIM_START, SIM_END,
    new_uuid, random_timestamp, settlement_delay,
    channel_for_account_type, realistic_amount, build_metadata,
    make_customer, make_account,
)
from data_simulator.scenarios.layering           import generate_layering_cluster
from data_simulator.scenarios.circular           import generate_circular_cluster
from data_simulator.scenarios.structuring        import generate_structuring_cluster
from data_simulator.scenarios.dormant_activation import generate_dormant_cluster
from data_simulator.scenarios.profile_mismatch_gen import generate_profile_mismatch_cluster

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

random.seed(42)
np.random.seed(42)

TOTAL_TRANSACTIONS = 100_000
FRAUD_PER_TYPOLOGY = 1_000
FRAUD_TOTAL        = FRAUD_PER_TYPOLOGY * 5
CLEAN_TOTAL        = TOTAL_TRANSACTIONS - FRAUD_TOTAL


def _make_clean_transaction(accounts: list[Account]) -> Transaction:
    src = random.choice(accounts)
    dst = random.choice(accounts)
    while dst.id == src.id or dst.is_dormant:
        dst = random.choice(accounts)
    channel   = channel_for_account_type(src.account_type)
    initiated = random_timestamp(SIM_START, SIM_END)
    settled   = settlement_delay(channel, initiated)
    amount    = realistic_amount(src.kyc_risk_tier)
    return Transaction(
        id=new_uuid(),
        reference_number=f"{channel}{initiated.strftime('%Y%m%d')}{random.randint(100000,999999)}",
        source_account_id=src.id,
        dest_account_id=dst.id,
        amount=amount,
        channel=channel,
        initiated_at=initiated,
        settled_at=settled,
        branch_id=src.branch_id,
        status="SETTLED",
        metadata=build_metadata(channel),
        is_fraud=False,
        fraud_type=None,
        fraud_cluster_id=None,
    )


def generate_universe(n_accounts: int = 2_000):
    log.info("Building universe (%d accounts)...", n_accounts)
    customers, accounts = [], []
    for _ in range(n_accounts):
        cust = make_customer()
        acct = make_account(cust)
        customers.append(cust)
        accounts.append(acct)

    active = [a for a in accounts if not a.is_dormant]
    log.info("Universe: %d accounts | %d active | %d dormant",
             len(accounts), len(active), len(accounts) - len(active))

    log.info("Injecting fraud scenarios...")
    txns_l, al, cl = generate_layering_cluster(n_clusters=FRAUD_PER_TYPOLOGY // 5,  shared_pool=active, sim_end=SIM_END)
    txns_c, ac, cc = generate_circular_cluster(n_clusters=FRAUD_PER_TYPOLOGY // 4,  shared_pool=active, sim_end=SIM_END)
    txns_s, as_, cs = generate_structuring_cluster(n_clusters=FRAUD_PER_TYPOLOGY // 3, shared_pool=active, sim_end=SIM_END)
    txns_d, ad, cd = generate_dormant_cluster(n_clusters=FRAUD_PER_TYPOLOGY // 4,   sim_end=SIM_END)
    txns_p, ap, cp = generate_profile_mismatch_cluster(n_clusters=FRAUD_PER_TYPOLOGY // 5, sim_end=SIM_END)

    for al_ in [al, ac, as_, ad, ap]: accounts.extend(al_)
    for cl_ in [cl, cc, cs, cd, cp]: customers.extend(cl_)
    active = [a for a in accounts if not a.is_dormant]

    fraud_txns = txns_l + txns_c + txns_s + txns_d + txns_p
    log.info("Fraud transactions: %d", len(fraud_txns))

    clean_count = TOTAL_TRANSACTIONS - len(fraud_txns)
    log.info("Generating %d clean transactions...", clean_count)
    all_txns = []
    for i in range(clean_count):
        if i % 10_000 == 0 and i > 0:
            log.info("  clean: %d / %d", i, clean_count)
        all_txns.append(_make_clean_transaction(active))

    all_txns.extend(fraud_txns)
    random.shuffle(all_txns)
    log.info("Total: %d transactions", len(all_txns))
    return customers, accounts, all_txns


def run(output_dir: str = "data", total_override: Optional[int] = None):
    global TOTAL_TRANSACTIONS, CLEAN_TOTAL
    if total_override:
        TOTAL_TRANSACTIONS = total_override
        CLEAN_TOTAL        = TOTAL_TRANSACTIONS - FRAUD_TOTAL

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    customers, accounts, transactions = generate_universe()

    log.info("Serialising to parquet...")

    tx_df = pd.DataFrame([{
        "id": t.id, "reference_number": t.reference_number,
        "source_account_id": t.source_account_id, "dest_account_id": t.dest_account_id,
        "amount": t.amount, "channel": t.channel,
        "initiated_at": t.initiated_at.isoformat(),
        "settled_at": t.settled_at.isoformat() if t.settled_at else None,
        "branch_id": t.branch_id, "status": t.status, "metadata": str(t.metadata),
        "is_fraud": t.is_fraud, "fraud_type": t.fraud_type, "fraud_cluster_id": t.fraud_cluster_id,
    } for t in transactions])

    acct_df = pd.DataFrame([{
        "id": a.id, "customer_id": a.customer_id, "account_type": a.account_type,
        "kyc_risk_tier": a.kyc_risk_tier, "declared_monthly_income": a.declared_monthly_income,
        "declared_occupation": a.declared_occupation, "open_date": a.open_date,
        "last_transaction_date": a.last_transaction_date, "is_dormant": a.is_dormant,
        "dormant_since": a.dormant_since, "status": a.status, "branch_id": a.branch_id,
    } for a in accounts])

    cust_df = pd.DataFrame([{
        "id": c.id, "name": c.name, "kyc_id": c.kyc_id, "risk_tier": c.risk_tier,
        "city": c.city, "state": c.state, "occupation": c.occupation,
        "segment": c.segment, "declared_monthly_income": c.declared_monthly_income,
    } for c in customers])

    labels_df = tx_df[["id", "is_fraud", "fraud_type", "fraud_cluster_id"]].copy()

    branch_ids = list({a.branch_id for a in accounts})
    nodes = (
        [{"node_type": "Account", "id": a.id, "account_type": a.account_type,
          "branch_id": a.branch_id, "kyc_risk_tier": a.kyc_risk_tier,
          "is_dormant": a.is_dormant, "status": a.status, "anoma_score": 0.0}
         for a in accounts] +
        [{"node_type": "Customer", "id": c.id, "name": c.name, "kyc_id": c.kyc_id,
          "risk_tier": c.risk_tier, "city": c.city, "occupation": c.occupation,
          "segment": c.segment}
         for c in customers] +
        [{"node_type": "Branch", "id": bid, "ifsc": bid} for bid in branch_ids]
    )
    edges = (
        [{"edge_type": "TRANSFERRED_TO", "source": t.source_account_id,
          "target": t.dest_account_id, "tx_id": t.id, "amount": t.amount,
          "timestamp": t.initiated_at.isoformat(), "channel": t.channel,
          "branch_id": t.branch_id, "is_fraud": t.is_fraud, "fraud_type": t.fraud_type}
         for t in transactions] +
        [{"edge_type": "OWNS", "source": a.customer_id, "target": a.id,
          "tx_id": None, "amount": None, "timestamp": None, "channel": None,
          "branch_id": None, "is_fraud": False, "fraud_type": None}
         for a in accounts] +
        [{"edge_type": "BELONGS_TO", "source": a.id, "target": a.branch_id,
          "tx_id": None, "amount": None, "timestamp": None, "channel": None,
          "branch_id": None, "is_fraud": False, "fraud_type": None}
         for a in accounts]
    )

    tx_df.to_parquet(out / "transactions.parquet",   index=False)
    acct_df.to_parquet(out / "accounts.parquet",     index=False)
    cust_df.to_parquet(out / "customers.parquet",    index=False)
    labels_df.to_parquet(out / "labels.parquet",     index=False)
    pd.DataFrame(nodes).to_parquet(out / "neo4j_nodes.parquet", index=False)
    pd.DataFrame(edges).to_parquet(out / "neo4j_edges.parquet", index=False)

    fraud_counts = tx_df[tx_df["is_fraud"]]["fraud_type"].value_counts()
    log.info("=" * 60)
    log.info("SIMULATION COMPLETE — %d transactions", len(tx_df))
    for ftype, cnt in fraud_counts.items():
        log.info("  %-30s %d", ftype, cnt)
    log.info("Output: %s", out.resolve())
    log.info("=" * 60)
    return tx_df, acct_df, cust_df, labels_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count",  type=int, default=None)
    parser.add_argument("--output", type=str, default="data")
    args = parser.parse_args()
    run(output_dir=args.output, total_override=args.count)