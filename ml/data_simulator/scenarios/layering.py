"""
ml/data_simulator/scenarios/layering.py

Generates LAYERING fraud clusters.

Pattern: ₹50L+ arrives at an account, fans out to 5-8 mule accounts within
90 minutes across multiple branches, each holding the money < 15 minutes
before forwarding onward. Classic Phase-2 money laundering.

Detection signals embedded:
  - fan-out degree > 5 within 1 hour
  - residency time < 15 minutes per hop
  - cross-branch (every hop uses a different branch IFSC)
  - off-hours preference (2–5 AM)
  - first-time counterparty relationships
"""

from __future__ import annotations

import random
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np

from data_simulator.simulator import (
    Account, Customer, Transaction,
    _new_uuid, _new_account_number, _random_ifsc,
    _make_customer, _make_account,
    IFSC_PREFIXES, SIM_END,
)

random.seed(None)  # each call gets fresh randomness within the seeded universe


def _off_hours_timestamp(base: datetime, window_hours: int = 3) -> datetime:
    """Return a timestamp between 02:00–05:00 local approximate."""
    # Push base to 2 AM of that day, add random minutes
    day_start = base.replace(hour=2, minute=0, second=0, microsecond=0)
    offset = timedelta(minutes=random.randint(0, window_hours * 60))
    return day_start + offset


def _make_layering_accounts(n_mules: int) -> tuple[list[Account], list[Customer]]:
    """Create one source account + n_mule intermediate accounts, each on a different branch."""
    accounts = []
    customers = []

    # Source account — medium KYC to look legitimate
    src_cust = _make_customer(kyc_tier="MEDIUM")
    src_acct = _make_account(src_cust, open_days_ago=random.randint(180, 730))
    src_acct.branch_id = _random_ifsc()
    customers.append(src_cust)
    accounts.append(src_acct)

    # Mule accounts — low KYC, different branches from source and from each other
    used_branches = {src_acct.branch_id}
    for _ in range(n_mules):
        mule_cust = _make_customer(kyc_tier="LOW")
        mule_acct = _make_account(mule_cust, open_days_ago=random.randint(30, 180))
        # Force a branch not yet used in this cluster
        while mule_acct.branch_id in used_branches:
            mule_acct.branch_id = _random_ifsc()
        used_branches.add(mule_acct.branch_id)
        customers.append(mule_cust)
        accounts.append(mule_acct)

    return accounts, customers


def generate_layering_cluster(
    n_clusters: int,
    shared_pool: list[Account],
    sim_end: datetime,
) -> tuple[list[Transaction], list[Account], list[Customer]]:
    """
    Generate `n_clusters` layering fraud incidents.
    Each cluster: 1 source → 5-8 mule accounts within ≤90 minutes.
    Returns (transactions, new_accounts, new_customers).
    """
    all_txns:     list[Transaction] = []
    all_accounts: list[Account]     = []
    all_customers: list[Customer]   = []

    for _ in range(n_clusters):
        cluster_id = _new_uuid()
        n_mules    = random.randint(5, 8)

        accounts, customers = _make_layering_accounts(n_mules)
        src_acct = accounts[0]
        mule_accounts = accounts[1:]

        all_accounts.extend(accounts)
        all_customers.extend(customers)

        # Anchor time — off hours, random day within simulation window
        days_ago = random.randint(1, 89)
        anchor_dt = (sim_end - timedelta(days=days_ago))
        anchor_dt = _off_hours_timestamp(anchor_dt)

        # Total dirty money in this cluster: 30L – 2Cr
        dirty_total = round(random.uniform(30_00_000, 2_00_00_000), 2)

        # ── Step 1: initial deposit into source account ──────────────────────
        initial_channel = random.choice(["NEFT", "RTGS", "SWIFT"])
        initial_tx = Transaction(
            id=_new_uuid(),
            reference_number=f"{initial_channel}{anchor_dt.strftime('%Y%m%d')}{random.randint(100000,999999)}",
            source_account_id=random.choice(shared_pool).id,  # unknown external origin
            dest_account_id=src_acct.id,
            amount=dirty_total,
            channel=initial_channel,
            initiated_at=anchor_dt - timedelta(minutes=random.randint(10, 60)),
            settled_at=anchor_dt,
            branch_id=src_acct.branch_id,
            status="SETTLED",
            metadata={"utr": f"UTR{random.randint(10**13, 10**14 - 1)}"},
            is_fraud=True,
            fraud_type="LAYERING",
            fraud_cluster_id=cluster_id,
        )
        all_txns.append(initial_tx)

        # ── Step 2: fan-out from source to mules ─────────────────────────────
        # Money leaves source within 90 minutes of arrival — residency < 15 min for each mule
        fan_out_start = anchor_dt + timedelta(minutes=random.randint(2, 14))

        # Split the dirty total across mules (approximately)
        splits = np.random.dirichlet(np.ones(n_mules)) * dirty_total
        splits = [round(float(s), 2) for s in splits]

        for i, (mule, split_amount) in enumerate(zip(mule_accounts, splits)):
            # Each fan-out leg happens within 90 minutes of the initial deposit
            fan_offset = timedelta(minutes=random.randint(i * 2, min(90, i * 10 + 15)))
            fan_time   = fan_out_start + fan_offset

            fan_tx = Transaction(
                id=_new_uuid(),
                reference_number=f"IMPS{fan_time.strftime('%Y%m%d')}{random.randint(100000,999999)}",
                source_account_id=src_acct.id,
                dest_account_id=mule.id,
                amount=split_amount,
                channel=random.choice(["IMPS", "NEFT", "UPI"]),
                initiated_at=fan_time,
                settled_at=fan_time + timedelta(seconds=random.randint(5, 120)),
                branch_id=src_acct.branch_id,
                status="SETTLED",
                metadata={"upi_txn_id": f"UPI{random.randint(10**15, 10**16-1)}"}
                         if random.random() < 0.5 else
                         {"utr": f"UTR{random.randint(10**13, 10**14-1)}"},
                is_fraud=True,
                fraud_type="LAYERING",
                fraud_cluster_id=cluster_id,
            )
            all_txns.append(fan_tx)

            # ── Step 3: each mule immediately forwards to the shared_pool (reconsolidation) ──
            # residency: 3–14 minutes
            residency_min = random.randint(3, 14)
            forward_time  = fan_time + timedelta(minutes=residency_min)

            reconsolidate_dest = random.choice(shared_pool)
            forward_tx = Transaction(
                id=_new_uuid(),
                reference_number=f"NEFT{forward_time.strftime('%Y%m%d')}{random.randint(100000,999999)}",
                source_account_id=mule.id,
                dest_account_id=reconsolidate_dest.id,
                amount=round(split_amount * random.uniform(0.90, 0.99), 2),  # slight fee deduction
                channel=random.choice(["NEFT", "RTGS"]),
                initiated_at=forward_time,
                settled_at=forward_time + timedelta(minutes=random.randint(1, 30)),
                branch_id=mule.branch_id,
                status="SETTLED",
                metadata={"utr": f"UTR{random.randint(10**13, 10**14-1)}"},
                is_fraud=True,
                fraud_type="LAYERING",
                fraud_cluster_id=cluster_id,
            )
            all_txns.append(forward_tx)

    return all_txns, all_accounts, all_customers