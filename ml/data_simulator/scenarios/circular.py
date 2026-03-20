"""
ml/data_simulator/scenarios/circular.py

Generates CIRCULAR / ROUND-TRIPPING fraud clusters.

Pattern: A → B → C → A. Money completes a directed cycle (length 2–7 hops),
returning to origin within 72 hours. Amounts stay within ±15% across the cycle
(accounting for fake fees). At least 2 first-time counterparty relationships.

This is trivially invisible in a flat transaction table but lights up instantly
in the Neo4j graph — Johnson's Algorithm finds it in milliseconds.

Detection signals embedded:
  - directed cycle of length 2–7
  - cycle completed within 72 hours
  - edge amounts within ±15% variance across the cycle
  - ≥2 first-time counterparty relationships in the cycle
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


def _make_circular_ring(length: int) -> tuple[list[Account], list[Customer]]:
    """
    Create `length` accounts that will form a ring: accts[0] → accts[1] → ... → accts[0].
    Each account is on a distinct branch.
    """
    accounts  = []
    customers = []
    used_branches = set()

    for i in range(length):
        # Mix KYC tiers: first node medium, others random low/medium
        tier = "MEDIUM" if i == 0 else random.choice(["LOW", "MEDIUM"])
        cust = make_customer(kyc_tier=tier)
        acct = make_account(cust, open_days_ago=random.randint(90, 1095))
        while acct.branch_id in used_branches:
            acct.branch_id = random_ifsc()
        used_branches.add(acct.branch_id)
        customers.append(cust)
        accounts.append(acct)

    return accounts, customers


def generate_circular_cluster(
    n_clusters: int,
    shared_pool: list[Account],
    sim_end: datetime,
) -> tuple[list[Transaction], list[Account], list[Customer]]:
    """
    Generate `n_clusters` circular fraud incidents.
    Each cluster: a ring of 2–7 accounts, money cycling back to origin within 72h.
    Returns (transactions, new_accounts, new_customers).
    """
    all_txns:      list[Transaction] = []
    all_accounts:  list[Account]     = []
    all_customers: list[Customer]    = []

    for _ in range(n_clusters):
        cluster_id  = new_uuid()
        ring_length = random.randint(2, 7)

        accounts, customers = _make_circular_ring(ring_length)
        all_accounts.extend(accounts)
        all_customers.extend(customers)

        # Anchor: random day in simulation window
        days_ago   = random.randint(1, 89)
        anchor_dt  = sim_end - timedelta(days=days_ago)
        anchor_dt  = anchor_dt.replace(
            hour=random.randint(9, 22),
            minute=random.randint(0, 59),
            second=random.randint(0, 59),
            microsecond=0,
        )

        # Starting amount: 20L – 1Cr (round-trip amount)
        start_amount = round(random.uniform(20_00_000, 1_00_00_000), 2)

        # Total time budget for the full cycle: 2–72 hours
        cycle_hours   = random.uniform(2, 72)
        # Divide time budget across legs — roughly equal spacing with jitter
        avg_leg_hours = cycle_hours / ring_length
        leg_times     = []
        t = anchor_dt
        for i in range(ring_length):
            leg_times.append(t)
            jitter = random.uniform(0.6, 1.4) * avg_leg_hours
            t = t + timedelta(hours=jitter)

        # Amount decays slightly each hop (fake fees: 0.5%–2% per hop)
        current_amount = start_amount
        for i in range(ring_length):
            src_acct = accounts[i]
            dst_acct = accounts[(i + 1) % ring_length]  # wraps back to accounts[0]

            channel = random.choice(["NEFT", "RTGS", "IMPS", "UPI"])
            leg_time = leg_times[i]
            settled  = leg_time + timedelta(minutes=random.randint(1, 60))

            fee_pct      = random.uniform(0.005, 0.02)
            tx_amount    = round(current_amount * (1 - fee_pct), 2)
            current_amount = tx_amount  # next leg starts with slightly less

            meta: dict = {}
            if channel in ("NEFT", "RTGS", "IMPS"):
                meta["utr"] = f"UTR{random.randint(10**13, 10**14-1)}"
            elif channel == "UPI":
                meta["upi_txn_id"] = f"UPI{random.randint(10**15, 10**16-1)}"

            txn = Transaction(
                id=new_uuid(),
                reference_number=f"{channel}{leg_time.strftime('%Y%m%d')}{random.randint(100000,999999)}",
                source_account_id=src_acct.id,
                dest_account_id=dst_acct.id,
                amount=tx_amount,
                channel=channel,
                initiated_at=leg_time,
                settled_at=settled,
                branch_id=src_acct.branch_id,
                status="SETTLED",
                metadata=meta,
                is_fraud=True,
                fraud_type="CIRCULAR",
                fraud_cluster_id=cluster_id,
            )
            all_txns.append(txn)

    return all_txns, all_accounts, all_customers
