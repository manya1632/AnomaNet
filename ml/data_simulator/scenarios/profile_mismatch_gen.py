"""
ml/data_simulator/scenarios/profile_mismatch_gen.py

Generates CUSTOMER PROFILE MISMATCH fraud clusters.

Pattern: Account holder's KYC declares a modest economic profile (kirana
shop owner, ₹40k/month) but the account suddenly processes crores of rupees
via channels inconsistent with their declared profile (e.g., a rural savings
account receiving SWIFT transfers from offshore entities).

Two sub-patterns:
  A) Volume mismatch: monthly volume >> 15× declared income
  B) Channel mismatch: rural SAVINGS account using SWIFT/RTGS at scale

Detection signals embedded:
  - monthly_volume > 15× declared_monthly_income
  - new counterparties outside the entire historical network
  - channel mismatch (SAVINGS account + SWIFT)
  - sudden shift in transaction time patterns
  - burst of large transactions after months of tiny ones
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


def _make_mismatch_account() -> tuple[Account, Customer, list[Transaction]]:
    """
    Create an account with:
      - Low declared income and occupation
      - Months of small, clean historical transactions (baseline)
    Returns (account, customer, historical_clean_txns).
    """
    cust = make_customer(kyc_tier="LOW")
    # Force a very low declared income
    cust.declared_monthly_income = round(random.uniform(20_000, 50_000), 2)
    cust.occupation = random.choice([
        "Kirana Shop Owner", "Vegetable Vendor", "Auto Driver",
        "Daily Wage Worker", "Retired Government Pensioner",
    ])
    acct = make_account(cust, open_days_ago=random.randint(365, 1825))  # 1–5 year old account
    acct.account_type = "SAVINGS"  # rural/retail savings account
    acct.kyc_risk_tier = "LOW"
    acct.declared_monthly_income = cust.declared_monthly_income

    # Generate 3–6 months of historical clean transactions (tiny amounts)
    historical_txns: list[Transaction] = []
    n_historical_months = random.randint(3, 6)
    for m in range(n_historical_months):
        n_txns_this_month = random.randint(2, 8)
        month_start = SIM_END - timedelta(days=(n_historical_months - m) * 30)
        for _ in range(n_txns_this_month):
            tx_time = month_start + timedelta(
                days=random.randint(0, 29),
                hours=random.randint(9, 17),
            )
            amount = round(random.uniform(500, cust.declared_monthly_income * 0.3), 2)
            htx = Transaction(
                id=new_uuid(),
                reference_number=f"UPI{tx_time.strftime('%Y%m%d')}{random.randint(100000,999999)}",
                source_account_id=acct.id,
                dest_account_id=new_account_number(),   # local grocery, utility, etc.
                amount=amount,
                channel="UPI",
                initiated_at=tx_time,
                settled_at=tx_time + timedelta(seconds=random.randint(2, 30)),
                branch_id=acct.branch_id,
                status="SETTLED",
                metadata={"upi_txn_id": f"UPI{random.randint(10**15, 10**16-1)}"},
                is_fraud=False,
                fraud_type=None,
                fraud_cluster_id=None,
            )
            historical_txns.append(htx)

    return acct, cust, historical_txns


def generate_profile_mismatch_cluster(
    n_clusters: int,
    sim_end: datetime,
) -> tuple[list[Transaction], list[Account], list[Customer]]:
    """
    Generate `n_clusters` profile mismatch fraud incidents.
    Each cluster:
      1. Account with tiny historical activity
      2. Sudden burst of large anomalous transactions in current month
         (volume >> 15× declared income, wrong channels)
    Returns (transactions, new_accounts, new_customers).
    """
    all_txns:      list[Transaction] = []
    all_accounts:  list[Account]     = []
    all_customers: list[Customer]    = []

    for _ in range(n_clusters):
        cluster_id = new_uuid()

        acct, cust, historical_txns = _make_mismatch_account()
        all_accounts.append(acct)
        all_customers.append(cust)

        # Historical clean transactions contribute to the dataset (not labelled fraud)
        all_txns.extend(historical_txns)

        # Burst window: current month (last 30 days)
        burst_start = sim_end - timedelta(days=30)

        # How much do they process? 15×–80× declared monthly income
        monthly_declared    = cust.declared_monthly_income
        burst_multiplier    = random.uniform(15, 80)
        total_burst_volume  = round(monthly_declared * burst_multiplier, 2)

        # Number of anomalous transactions
        n_burst_txns = random.randint(5, 20)

        # Channel mismatch: SWIFT for a SAVINGS account is a huge red flag
        anomalous_channels = random.choices(
            ["SWIFT", "RTGS", "NEFT"],
            weights=[0.40, 0.35, 0.25],
            k=n_burst_txns,
        )

        # Split volume across burst transactions
        import numpy as np
        splits = np.random.dirichlet(np.ones(n_burst_txns)) * total_burst_volume
        splits = [round(float(s), 2) for s in splits]

        for i in range(n_burst_txns):
            tx_time = burst_start + timedelta(
                hours=random.randint(0, 24 * 30),
            )
            # Cap at sim_end
            if tx_time > sim_end:
                tx_time = sim_end - timedelta(hours=random.randint(1, 48))

            channel = anomalous_channels[i]
            amount  = splits[i]

            meta: dict = {}
            if channel == "SWIFT":
                meta["swift_ref"]      = f"SWIFT{random.randint(10**7, 10**8-1)}"
                meta["originator_bic"] = f"BNPP{random.choice(['FRPP', 'DEFF', 'GBLO', 'SGSG'])}"
            elif channel in ("RTGS", "NEFT"):
                meta["utr"] = f"UTR{random.randint(10**13, 10**14-1)}"

            burst_tx = Transaction(
                id=new_uuid(),
                reference_number=f"{channel}{tx_time.strftime('%Y%m%d')}{random.randint(100000,999999)}",
                # Inbound: suspicious foreign or corporate accounts
                source_account_id=new_account_number(),
                dest_account_id=acct.id,
                amount=amount,
                channel=channel,
                initiated_at=tx_time,
                settled_at=tx_time + timedelta(
                    days=1 if channel == "SWIFT" else 0,
                    hours=random.randint(0, 4),
                ),
                branch_id=acct.branch_id,
                status="SETTLED",
                metadata=meta,
                is_fraud=True,
                fraud_type="PROFILE_MISMATCH",
                fraud_cluster_id=cluster_id,
            )
            all_txns.append(burst_tx)

    return all_txns, all_accounts, all_customers
