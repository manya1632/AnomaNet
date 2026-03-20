"""
ml/data_simulator/models.py

Shared data classes, constants, and pure helper functions.
Imported by BOTH simulator.py and all scenario scripts.
This file imports NOTHING from the data_simulator package — no circular risk.
"""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

from faker import Faker

fake = Faker("en_IN")

# ── Simulation window ────────────────────────────────────────────────────────
SIM_END   = datetime.now(tz=timezone.utc).replace(microsecond=0)
SIM_START = SIM_END - timedelta(days=90)

# ── Constants ────────────────────────────────────────────────────────────────
CHANNELS        = ["UPI", "NEFT", "IMPS", "RTGS", "CASH", "BRANCH", "SWIFT"]
CHANNEL_WEIGHTS = [0.42, 0.22, 0.18, 0.08, 0.05, 0.03, 0.02]

IFSC_PREFIXES = [
    "HDFC0001", "HDFC0002", "HDFC0003",
    "ICIC0001", "ICIC0002",
    "SBIN0001", "SBIN0002", "SBIN0003",
    "AXIS0001", "AXIS0002",
    "KKBK0001", "PUNB0001",
]

ACCOUNT_TYPES        = ["SAVINGS", "CURRENT", "OD", "LOAN", "NRE", "NRO"]
ACCOUNT_TYPE_WEIGHTS = [0.55, 0.25, 0.08, 0.05, 0.04, 0.03]

KYC_TIERS        = ["LOW", "MEDIUM", "HIGH", "PEP"]
KYC_TIER_WEIGHTS = [0.50, 0.35, 0.13, 0.02]

OCCUPATIONS = [
    "Salaried Employee", "Business Owner", "Self Employed Professional",
    "Government Employee", "Retired", "Student", "Agriculturist",
    "Trader", "Freelancer", "Homemaker",
]

INCOME_RANGES = {
    "LOW":    (8_000,     40_000),
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
    is_fraud: bool = False
    fraud_type: Optional[str] = None
    fraud_cluster_id: Optional[str] = None


# ── Pure helper functions ─────────────────────────────────────────────────────

def new_uuid() -> str:
    return str(uuid.uuid4())


def new_account_number() -> str:
    return str(random.randint(100_000_000_000, 999_999_999_999))


def random_ifsc() -> str:
    prefix = random.choice(IFSC_PREFIXES)
    suffix = f"{random.randint(0, 9999):04d}"
    return f"{prefix}{suffix}"


def random_timestamp(start: datetime, end: datetime) -> datetime:
    delta = end - start
    return start + timedelta(seconds=random.randint(0, int(delta.total_seconds())))


def settlement_delay(channel: str, initiated_at: datetime) -> Optional[datetime]:
    delays = {
        "UPI":    timedelta(seconds=random.randint(1, 30)),
        "IMPS":   timedelta(seconds=random.randint(5, 120)),
        "NEFT":   timedelta(hours=random.choice([0, 2, 4, 6])),
        "RTGS":   timedelta(minutes=random.randint(15, 90)),
        "CASH":   None,
        "BRANCH": timedelta(hours=random.randint(1, 4)),
        "SWIFT":  timedelta(days=random.randint(1, 3)),
    }
    delay = delays.get(channel)
    return (initiated_at + delay) if delay else None


def channel_for_account_type(account_type: str) -> str:
    if account_type in ("NRE", "NRO"):
        return random.choices(["SWIFT", "NEFT", "RTGS"], weights=[0.5, 0.3, 0.2])[0]
    if account_type == "CURRENT":
        return random.choices(CHANNELS, weights=[0.2, 0.3, 0.15, 0.25, 0.05, 0.03, 0.02])[0]
    return random.choices(CHANNELS, weights=CHANNEL_WEIGHTS)[0]


def realistic_amount(kyc_tier: str) -> float:
    low, high = INCOME_RANGES[kyc_tier]
    if random.random() < 0.70:
        amount = random.uniform(low * 0.01, low * 0.5)
    elif random.random() < 0.90:
        amount = random.uniform(low * 0.5, high * 0.3)
    else:
        amount = random.uniform(high * 0.3, high * 1.2)
    return round(amount, 2)


def build_metadata(channel: str) -> dict:
    meta: dict = {}
    if channel in ("NEFT", "RTGS", "IMPS"):
        meta["utr"] = f"UTR{random.randint(10**13, 10**14-1)}"
    if channel == "SWIFT":
        meta["swift_ref"] = f"SWIFT{random.randint(10_000_000, 99_999_999)}"
    if channel == "UPI":
        meta["upi_txn_id"] = f"UPI{random.randint(10**15, 10**16 - 1)}"
        meta["device_id"]  = str(uuid.uuid4())
    return meta


def make_customer(kyc_tier: Optional[str] = None) -> Customer:
    tier = kyc_tier or random.choices(KYC_TIERS, weights=KYC_TIER_WEIGHTS)[0]
    income_lo, income_hi = INCOME_RANGES[tier]
    return Customer(
        id=new_uuid(),
        name=fake.name(),
        kyc_id=(
            f"PAN"
            f"{''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=5))}"
            f"{random.randint(1000, 9999)}"
            f"{''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=1))}"
        ),
        risk_tier=tier,
        city=fake.city(),
        state=fake.state(),
        occupation=random.choice(OCCUPATIONS),
        segment="RETAIL" if tier in ("LOW", "MEDIUM") else "WEALTH",
        declared_monthly_income=round(random.uniform(income_lo, income_hi), 2),
    )


def make_account(
    customer: Customer,
    open_days_ago: Optional[int] = None,
    force_dormant: bool = False,
) -> Account:
    if open_days_ago is None:
        open_days_ago = random.randint(180, 3650)
    open_date_dt = SIM_END - timedelta(days=open_days_ago)

    is_dormant    = False
    dormant_since = None
    last_tx_date  = None

    if force_dormant:
        dormant_days     = random.randint(420, 730)
        dormant_since_dt = SIM_END - timedelta(days=dormant_days)
        dormant_since    = dormant_since_dt.strftime("%Y-%m-%d")
        last_tx_date     = dormant_since
        is_dormant       = True
    elif random.random() < 0.06:
        dormant_days     = random.randint(365, 1000)
        dormant_since_dt = SIM_END - timedelta(days=dormant_days)
        dormant_since    = dormant_since_dt.strftime("%Y-%m-%d")
        last_tx_date     = dormant_since
        is_dormant       = True
    else:
        last_tx_date = (SIM_END - timedelta(days=random.randint(0, 30))).strftime("%Y-%m-%d")

    return Account(
        id=new_account_number(),
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
        branch_id=random_ifsc(),
    )