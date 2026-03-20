"""
ml/data_simulator/scenarios/dormant_activation.py

Generates DORMANT ACCOUNT SUDDEN ACTIVATION fraud clusters.

Pattern: An account that has been silent for 14–24 months suddenly receives
a large transfer (₹50L–₹5Cr) and wires it back out within 6 hours to one
or more accounts in the fraud network. Classic profile: identity-theft
takeover or a pre-planted account created for a single fraud event.

Detection signals embedded:
  - account.is_dormant = True at the time of activation
  - first post-dormancy transaction exceeds 10× historical average
  - immediate outbound transfer (< 6 hours) to accounts in flagged network
  - new KYC details recently updated (modelled via metadata flag)
  - account has been dormant 14–24 months (420–730 days)
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


def _make_dormant_account() -> tuple[Account, Customer]:
    """Create an account that has been dormant for 14–24 months."""
    # Low KYC — these are often identity-theft victims or shell accounts
    cust = make_customer(kyc_tier=random.choice(["LOW", "MEDIUM"]))
    # Historical average for this account should be tiny (₹5k–₹50k)
    cust.declared_monthly_income = round(random.uniform(15_000, 45_000), 2)
    acct = make_account(cust, force_dormant=True)
    return acct, cust


def generate_dormant_cluster(
    n_clusters: int,
    sim_end: datetime,
) -> tuple[list[Transaction], list[Account], list[Customer]]:
    """
    Generate `n_clusters` dormant-activation fraud incidents.
    Each cluster:
      1. A dormant account reactivates
      2. Receives a large inbound transfer
      3. Wires money out within 2–6 hours to 1–3 accounts

    Returns (transactions, new_accounts, new_customers).
    """
    all_txns:      list[Transaction] = []
    all_accounts:  list[Account]     = []
    all_customers: list[Customer]    = []

    for _ in range(n_clusters):
        cluster_id = new_uuid()

        dormant_acct, dormant_cust = _make_dormant_account()
        all_accounts.append(dormant_acct)
        all_customers.append(dormant_cust)

        # Create 1–3 destination accounts (fraud network recipients)
        n_dest = random.randint(1, 3)
        dest_accounts  = []
        dest_customers = []
        for _ in range(n_dest):
            d_cust = make_customer(kyc_tier="LOW")
            d_acct = make_account(d_cust, open_days_ago=random.randint(30, 365))
            dest_accounts.append(d_acct)
            dest_customers.append(d_cust)
        all_accounts.extend(dest_accounts)
        all_customers.extend(dest_customers)

        # Activation timestamp: recent (last 45 days)
        days_ago      = random.randint(1, 45)
        activation_dt = sim_end - timedelta(days=days_ago)
        activation_dt = activation_dt.replace(
            hour=random.randint(8, 22),
            minute=random.randint(0, 59),
            second=random.randint(0, 59),
            microsecond=0,
        )

        # Large inbound amount — 10× or more the historical average
        # Historical average for these accounts: ₹5k–₹30k
        hist_avg        = round(random.uniform(5_000, 30_000), 2)
        inbound_amount  = round(hist_avg * random.uniform(10, 50) * random.uniform(10, 100), 2)
        # Cap at 5Cr for realism
        inbound_amount  = min(inbound_amount, 5_00_00_000)
        inbound_amount  = max(inbound_amount, 50_00_000)   # at least 50L

        # ── Inbound transfer (reactivation event) ────────────────────────────
        inbound_channel = random.choice(["NEFT", "RTGS", "SWIFT"])
        inbound_tx = Transaction(
            id=new_uuid(),
            reference_number=f"{inbound_channel}{activation_dt.strftime('%Y%m%d')}{random.randint(100000,999999)}",
            # Source: random external account — not in our modelled universe (unknown origin)
            source_account_id=new_account_number(),
            dest_account_id=dormant_acct.id,
            amount=inbound_amount,
            channel=inbound_channel,
            initiated_at=activation_dt - timedelta(hours=random.randint(1, 3)),
            settled_at=activation_dt,
            branch_id=dormant_acct.branch_id,
            status="SETTLED",
            metadata={
                "utr": f"UTR{random.randint(10**13, 10**14-1)}",
                "kyc_recently_updated": True,   # flag: KYC update before activation
                "new_mobile": True,
                "historical_avg_txn": hist_avg,
            },
            is_fraud=True,
            fraud_type="DORMANT_ACTIVATION",
            fraud_cluster_id=cluster_id,
        )
        all_txns.append(inbound_tx)

        # Update dormant account status in-memory (it's now active)
        dormant_acct.is_dormant = False
        dormant_acct.status     = "ACTIVE"
        dormant_acct.last_transaction_date = activation_dt.strftime("%Y-%m-%d")

        # ── Outbound transfers (within 2–6 hours of inbound) ─────────────────
        outbound_start    = activation_dt + timedelta(minutes=random.randint(30, 120))
        remaining_amount  = inbound_amount

        for i, dest_acct in enumerate(dest_accounts):
            # Last dest gets the remaining balance
            if i < len(dest_accounts) - 1:
                split = round(remaining_amount * random.uniform(0.30, 0.65), 2)
            else:
                split = remaining_amount

            remaining_amount -= split
            if split < 10_000:
                break   # too small to bother with

            out_offset  = timedelta(minutes=random.randint(i * 20, i * 60 + 90))
            out_time    = outbound_start + out_offset
            out_channel = random.choice(["NEFT", "RTGS", "IMPS"])

            outbound_tx = Transaction(
                id=new_uuid(),
                reference_number=f"{out_channel}{out_time.strftime('%Y%m%d')}{random.randint(100000,999999)}",
                source_account_id=dormant_acct.id,
                dest_account_id=dest_acct.id,
                amount=split,
                channel=out_channel,
                initiated_at=out_time,
                settled_at=out_time + timedelta(minutes=random.randint(1, 60)),
                branch_id=dormant_acct.branch_id,
                status="SETTLED",
                metadata={"utr": f"UTR{random.randint(10**13, 10**14-1)}"},
                is_fraud=True,
                fraud_type="DORMANT_ACTIVATION",
                fraud_cluster_id=cluster_id,
            )
            all_txns.append(outbound_tx)

    return all_txns, all_accounts, all_customers
