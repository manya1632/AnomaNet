"""
ml/data_simulator/scenarios/structuring.py

Generates STRUCTURING / SMURFING fraud clusters.

Pattern: A single entity (or coordinated group) makes 3–7 cash deposits
all clustered just below the ₹10 lakh CTR threshold within a short time
window (≤7 days), possibly across multiple branches. Individually each
deposit is below the regulatory radar; collectively they represent
unreported cash of ₹25L–₹65L.

Three threshold tiers modelled:
  - Tier 1: ₹10,00,000 (primary CTR threshold)
  - Tier 2: ₹5,00,000  (secondary SAR threshold)
  - Tier 3: ₹2,00,000  (tertiary)

Detection signals embedded:
  - 3+ transactions from same entity in 7-day window
  - amounts ∈ [threshold × 0.90, threshold × 0.985] — just below threshold
  - aggregate exceeds the threshold several times over
  - may span multiple branches (smurfing variant)
  - cash or branch channel
"""

from __future__ import annotations

import random
import uuid
from datetime import datetime, timedelta

from data_simulator.models import (
    Account, Customer, Transaction,
    new_uuid, new_account_number, random_ifsc,
    make_customer, make_account, SIM_END,
)

# CTR threshold tiers in INR
CTR_THRESHOLDS = [
    10_00_000,   # ₹10 lakhs (primary)
    5_00_000,    # ₹5 lakhs
    2_00_000,    # ₹2 lakhs
]
THRESHOLD_WEIGHTS = [0.70, 0.20, 0.10]


def _structuring_amount(threshold: int) -> float:
    """Return an amount clustered just below the given threshold."""
    # 90%–98.5% of threshold, avoiding round numbers
    pct    = random.uniform(0.900, 0.985)
    amount = threshold * pct
    # Add sub-rupee randomness to avoid suspiciously round numbers
    amount += random.uniform(-500, 500)
    return round(max(amount, threshold * 0.85), 2)


def generate_structuring_cluster(
    n_clusters: int,
    shared_pool: list[Account],
    sim_end: datetime,
) -> tuple[list[Transaction], list[Account], list[Customer]]:
    """
    Generate `n_clusters` structuring fraud incidents.
    Each cluster: 3–7 cash/branch transactions from same account,
    amounts just below a CTR threshold, within ≤7 days.
    Returns (transactions, new_accounts, new_customers).
    """
    all_txns:      list[Transaction] = []
    all_accounts:  list[Account]     = []
    all_customers: list[Customer]    = []

    for _ in range(n_clusters):
        cluster_id = new_uuid()
        threshold  = random.choices(CTR_THRESHOLDS, weights=THRESHOLD_WEIGHTS)[0]
        n_deposits = random.randint(3, 7)

        # The structuring account — low/medium KYC
        tier = random.choice(["LOW", "MEDIUM"])
        cust = make_customer(kyc_tier=tier)
        # Make declared income much lower than what they're depositing
        cust.declared_monthly_income = round(
            random.uniform(15_000, 60_000), 2
        )
        acct = make_account(cust, open_days_ago=random.randint(180, 1095))
        all_accounts.append(acct)
        all_customers.append(cust)

        # Smurfing variant: sometimes use multiple branches
        is_smurfing = random.random() < 0.35
        branch_pool = [acct.branch_id]
        if is_smurfing:
            for _ in range(n_deposits - 1):
                branch_pool.append(random_ifsc())

        # Window: random start day, deposits spread within 7 days
        days_ago     = random.randint(7, 89)
        window_start = sim_end - timedelta(days=days_ago)
        window_end   = window_start + timedelta(days=7)

        # Destination: money goes to a pooling account or the shared pool
        pooling_dest = random.choice(shared_pool)

        for i in range(n_deposits):
            # Spread deposits across the 7-day window — not all same day
            deposit_offset = timedelta(
                hours=random.randint(0, int((window_end - window_start).total_seconds() // 3600))
            )
            deposit_time = window_start + deposit_offset
            deposit_time = deposit_time.replace(
                hour=random.randint(9, 17),   # business hours — more natural
                minute=random.randint(0, 59),
                second=random.randint(0, 59),
            )

            branch = branch_pool[i % len(branch_pool)]
            amount = _structuring_amount(threshold)
            channel = random.choice(["CASH", "BRANCH"])

            meta: dict = {
                "teller_id": f"TLR{random.randint(1000, 9999)}",
                "counter": random.randint(1, 12),
            }

            deposit_tx = Transaction(
                id=new_uuid(),
                reference_number=f"CASH{deposit_time.strftime('%Y%m%d')}{random.randint(100000,999999)}",
                source_account_id=acct.id,
                dest_account_id=pooling_dest.id,
                amount=amount,
                channel=channel,
                initiated_at=deposit_time,
                settled_at=deposit_time + timedelta(minutes=random.randint(5, 30)),
                branch_id=branch,
                status="SETTLED",
                metadata=meta,
                is_fraud=True,
                fraud_type="STRUCTURING",
                fraud_cluster_id=cluster_id,
            )
            all_txns.append(deposit_tx)

    return all_txns, all_accounts, all_customers
