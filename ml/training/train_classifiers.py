"""
ml/training/train_classifiers.py

Trains all three classical ML models on the 100k simulator dataset:
  1. Isolation Forest    → core/models/isolation_forest_layering.pkl
  2. XGBoost             → core/models/xgboost_structuring.pkl
  3. Logistic Regression → core/models/logistic_dormancy.pkl

Usage:
  python -m training.train_classifiers --no-mlflow
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MODELS_DIR   = Path(__file__).parent.parent / "core" / "models"
MLFLOW_URI   = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
RANDOM_STATE = 42

CTR_THRESHOLDS = [10_00_000, 5_00_000, 2_00_000]
NEAR_LOW, NEAR_HIGH = 0.85, 0.99


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(data_dir: str = "data"):
    d = Path(data_dir)
    tx   = pd.read_parquet(d / "transactions.parquet")
    acct = pd.read_parquet(d / "accounts.parquet")
    cust = pd.read_parquet(d / "customers.parquet")
    tx["initiated_at"] = pd.to_datetime(tx["initiated_at"], format="ISO8601", utc=True)
    log.info("Loaded %d transactions | %d accounts | %d customers",
             len(tx), len(acct), len(cust))
    return tx, acct, cust


def save_model(model, filename: str):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / filename
    with open(path, "wb") as f:
        pickle.dump(model, f)
    log.info("Saved → %s", path)


# ── 1. Isolation Forest — Layering ───────────────────────────────────────────
# Fix: label only the SOURCE account of confirmed LAYERING transactions,
# not every account that ever appeared in a layering transaction.
# This keeps fraud rate realistic (~5%) so the Forest learns correctly.

def build_layering_features(tx: pd.DataFrame, acct: pd.DataFrame) -> pd.DataFrame:
    log.info("Building layering features...")
    tx = tx.copy()
    tx["is_off_hours"] = tx["initiated_at"].dt.hour.between(2, 5).astype(int)

    # Per-account velocity aggregations (all accounts, clean + fraud)
    agg = tx.groupby("source_account_id").agg(
        tx_count_total  =("id",             "count"),
        total_amount    =("amount",          "sum"),
        avg_amount      =("amount",          "mean"),
        std_amount      =("amount",          "std"),
        unique_dest     =("dest_account_id", "nunique"),
        unique_branches =("branch_id",       "nunique"),
        off_hours_ratio =("is_off_hours",    "mean"),
        channel_nunique =("channel",         "nunique"),
    ).reset_index().rename(columns={"source_account_id": "account_id"})

    agg["std_amount"]         = agg["std_amount"].fillna(0)
    agg["cross_branch_ratio"] = agg["unique_branches"] / agg["tx_count_total"].clip(lower=1)
    agg["counterparty_ratio"] = agg["unique_dest"]     / agg["tx_count_total"].clip(lower=1)

    # FIXED: only label the PRIMARY source account of layering clusters
    # (the account that fans out), not every mule that received money
    layering_sources = set(
        tx[tx["fraud_type"] == "LAYERING"]
        .groupby("source_account_id")["id"]
        .count()
        .where(lambda x: x >= 3)   # source account has 3+ outbound layering txns
        .dropna()
        .index
        .tolist()
    )
    agg["is_fraud"] = agg["account_id"].isin(layering_sources).astype(int)

    fraud_pct = agg["is_fraud"].mean() * 100
    log.info("Layering: %d accounts | fraud=%d (%.1f%%) | clean=%d",
             len(agg), agg["is_fraud"].sum(), fraud_pct, (agg["is_fraud"]==0).sum())
    return agg


def train_isolation_forest(features_df: pd.DataFrame, use_mlflow: bool = True) -> IsolationForest:
    feature_cols = [
        "tx_count_total", "total_amount", "avg_amount", "std_amount",
        "unique_dest", "unique_branches", "off_hours_ratio",
        "channel_nunique", "cross_branch_ratio", "counterparty_ratio",
    ]

    fraud_rate   = features_df["is_fraud"].mean()
    contamination = max(0.01, min(float(fraud_rate), 0.10))  # clamp 1%–10%
    log.info("Isolation Forest contamination=%.3f (fraud rate=%.3f)", contamination, fraud_rate)

    # Train on clean accounts only — model learns what "normal" looks like
    clean = features_df[features_df["is_fraud"] == 0][feature_cols].values
    all_X = features_df[feature_cols].values
    all_y = features_df["is_fraud"].values

    model = IsolationForest(
        n_estimators=300,
        max_samples="auto",
        contamination=contamination,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(clean)

    preds  = (model.predict(all_X) == -1).astype(int)
    scores = -model.decision_function(all_X)
    f1  = f1_score(all_y, preds, zero_division=0)
    p   = precision_score(all_y, preds, zero_division=0)
    r   = recall_score(all_y, preds, zero_division=0)
    try:
        auc = roc_auc_score(all_y, scores)
    except Exception:
        auc = 0.0

    log.info("Isolation Forest | F1=%.3f | P=%.3f | R=%.3f | AUC=%.3f", f1, p, r, auc)

    if use_mlflow:
        try:
            import mlflow, mlflow.sklearn
            mlflow.set_tracking_uri(MLFLOW_URI)
            mlflow.set_experiment("AnomaNet_Classifiers")
            with mlflow.start_run(run_name="IsolationForest_Layering"):
                mlflow.log_params({"n_estimators": 300, "contamination": contamination, "detector": "layering"})
                mlflow.log_metrics({"f1": f1, "precision": p, "recall": r, "auc": auc})
                mlflow.sklearn.log_model(model, "isolation_forest_layering")
        except Exception as e:
            log.warning("MLflow logging failed: %s", e)

    return model


# ── 2. XGBoost — Structuring ──────────────────────────────────────────────────
# Fix: include ALL accounts with cash transactions, not just near-threshold ones.
# Near-threshold accounts = fraud candidates; others = clean negatives.
# This gives XGBoost both classes to learn from.

def build_structuring_features(tx: pd.DataFrame, acct: pd.DataFrame) -> pd.DataFrame:
    log.info("Building structuring features...")

    cash_tx = tx[tx["channel"].isin(["CASH", "BRANCH"])].copy()

    if len(cash_tx) == 0:
        log.warning("No cash transactions found")
        return pd.DataFrame()

    acct_lookup = acct.set_index("id")["declared_monthly_income"].to_dict()
    rows = []

    for acct_id, group in cash_tx.groupby("source_account_id"):
        declared = float(acct_lookup.get(acct_id, 0) or 0)

        # Count near-threshold transactions
        near_count = 0
        best_t     = CTR_THRESHOLDS[0]
        pcts       = []
        for _, row in group.iterrows():
            amt = float(row["amount"])
            for t in CTR_THRESHOLDS:
                pct = amt / t
                if NEAR_LOW <= pct <= NEAR_HIGH:
                    near_count += 1
                    best_t = t
                    pcts.append(pct)
                    break

        ts     = group["initiated_at"].sort_values().tolist()
        deltas = [(ts[i]-ts[i-1]).total_seconds()/3600 for i in range(1, len(ts))]
        agg_amt    = float(group["amount"].sum())
        n_branches = int(group["branch_id"].nunique())

        # Label: account is fraud only if it has STRUCTURING transactions
        is_fraud = int(group["fraud_type"].eq("STRUCTURING").any())

        rows.append({
            "account_id":                acct_id,
            "n_txns_below_threshold_7d": near_count,
            "aggregate_amount_7d":       agg_amt,
            "n_distinct_branches_7d":    n_branches,
            "min_time_delta_hours":      min(deltas) if deltas else 0.0,
            "max_time_delta_hours":      max(deltas) if deltas else 0.0,
            "closest_threshold":         float(best_t),
            "avg_pct_below_threshold":   float(np.mean(pcts)) if pcts else 0.0,
            "is_cash_channel":           1.0,
            "declared_income_ratio":     agg_amt / declared if declared > 0 else 0.0,
            "is_fraud":                  is_fraud,
        })

    df = pd.DataFrame(rows)
    log.info("Structuring: %d accounts | fraud=%d | clean=%d",
             len(df), df["is_fraud"].sum(), (df["is_fraud"]==0).sum())
    return df


def train_xgboost(features_df: pd.DataFrame, use_mlflow: bool = True) -> XGBClassifier:
    # Guard: need both classes
    if features_df["is_fraud"].nunique() < 2:
        log.error("XGBoost needs both fraud and clean samples — only one class found")
        raise ValueError("Need at least one fraud and one clean sample")

    feature_cols = [
        "n_txns_below_threshold_7d", "aggregate_amount_7d",
        "n_distinct_branches_7d", "min_time_delta_hours",
        "max_time_delta_hours", "closest_threshold",
        "avg_pct_below_threshold", "is_cash_channel",
        "declared_income_ratio",
    ]
    X = features_df[feature_cols].values.astype(np.float32)
    y = features_df["is_fraud"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    scale_pos = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)
    log.info("XGBoost | train=%d | test=%d | scale_pos_weight=%.1f",
             len(X_train), len(X_test), scale_pos)

    model = XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        eval_metric="logloss",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    f1  = f1_score(y_test, preds, zero_division=0)
    p   = precision_score(y_test, preds, zero_division=0)
    r   = recall_score(y_test, preds, zero_division=0)
    auc = roc_auc_score(y_test, probs)

    log.info("XGBoost Structuring | F1=%.3f | P=%.3f | R=%.3f | AUC=%.3f", f1, p, r, auc)
    print(classification_report(y_test, preds, target_names=["clean", "structuring"]))

    if use_mlflow:
        try:
            import mlflow, mlflow.sklearn
            mlflow.set_tracking_uri(MLFLOW_URI)
            mlflow.set_experiment("AnomaNet_Classifiers")
            with mlflow.start_run(run_name="XGBoost_Structuring"):
                mlflow.log_params({"n_estimators": 300, "max_depth": 5, "detector": "structuring"})
                mlflow.log_metrics({"f1": f1, "precision": p, "recall": r, "auc": auc})
                mlflow.sklearn.log_model(model, "xgboost_structuring")
        except Exception as e:
            log.warning("MLflow logging failed: %s", e)

    return model


# ── 3. Logistic Regression — Dormancy ────────────────────────────────────────

def build_dormancy_features(tx: pd.DataFrame, acct: pd.DataFrame) -> pd.DataFrame:
    log.info("Building dormancy features...")

    # Include ALL accounts that have ever been dormant OR have dormant_since set
    dormant = acct[
        acct["is_dormant"].fillna(False) |
        acct["dormant_since"].notna()
    ].copy()

    if len(dormant) == 0:
        log.warning("No dormant accounts in dataset")
        return pd.DataFrame()

    rows = []
    for _, ar in dormant.iterrows():
        acct_id   = ar["id"]
        acct_txns = tx[
            (tx["source_account_id"] == acct_id) |
            (tx["dest_account_id"]   == acct_id)
        ].sort_values("initiated_at")

        if len(acct_txns) == 0:
            continue

        # Dormancy duration
        dormancy_months = 0.0
        dormant_since   = ar.get("dormant_since")
        if pd.notna(dormant_since):
            try:
                ds = pd.Timestamp(dormant_since, tz="UTC")
                dormancy_months = (
                    acct_txns["initiated_at"].max() - ds
                ).total_seconds() / (30.44 * 86400)
            except Exception:
                pass

        # First inbound transaction
        inbound = acct_txns[acct_txns["dest_account_id"] == acct_id]
        if len(inbound) == 0:
            continue

        first_inbound_amt  = float(inbound.iloc[0]["amount"])
        first_inbound_time = inbound.iloc[0]["initiated_at"]

        # Historical average before this transaction
        hist     = acct_txns[acct_txns["initiated_at"] < first_inbound_time]
        hist_avg = float(hist["amount"].mean()) if len(hist) > 0 else 1_000.0

        # Speed of first outbound after inbound
        out = acct_txns[
            (acct_txns["source_account_id"] == acct_id) &
            (acct_txns["initiated_at"] > first_inbound_time)
        ]
        outbound_hours = (
            (out.iloc[0]["initiated_at"] - first_inbound_time).total_seconds() / 3600
            if len(out) > 0 else 999.0
        )

        declared = float(ar.get("declared_monthly_income") or 0)
        kyc_tier = ar.get("kyc_risk_tier", "LOW")

        rows.append({
            "account_id":                      acct_id,
            "dormancy_duration_months":        max(dormancy_months, 0.0),
            "amount_vs_historical_avg_ratio":  first_inbound_amt / max(hist_avg, 1.0),
            "amount_vs_declared_income_ratio": first_inbound_amt / max(declared, 1.0),
            "post_activation_outbound_hours":  outbound_hours,
            "kyc_recently_updated":            0,
            "is_high_kyc_risk":                int(kyc_tier in ("HIGH", "PEP")),
            "inbound_amount_log":              float(np.log1p(first_inbound_amt)),
            "is_fraud": int(acct_txns["fraud_type"].eq("DORMANT_ACTIVATION").any()),
        })

    df = pd.DataFrame(rows)

    if len(df) == 0:
        return df

    log.info("Dormancy: %d accounts | fraud=%d | clean=%d",
             len(df), df["is_fraud"].sum(), (df["is_fraud"]==0).sum())
    return df


def train_logistic_regression(features_df: pd.DataFrame, use_mlflow: bool = True):
    if features_df["is_fraud"].nunique() < 2:
        log.warning("Logistic regression needs both classes — skipping")
        return None

    feature_cols = [
        "dormancy_duration_months", "amount_vs_historical_avg_ratio",
        "amount_vs_declared_income_ratio", "post_activation_outbound_hours",
        "kyc_recently_updated", "is_high_kyc_risk", "inbound_amount_log",
    ]
    X = features_df[feature_cols].fillna(0).values.astype(np.float32)
    y = features_df["is_fraud"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    log.info("Logistic Regression | train=%d | test=%d", len(X_train), len(X_test))

    model = LogisticRegression(
        class_weight="balanced", max_iter=1000,
        random_state=RANDOM_STATE, C=1.0,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    f1  = f1_score(y_test, preds, zero_division=0)
    p   = precision_score(y_test, preds, zero_division=0)
    r   = recall_score(y_test, preds, zero_division=0)
    try:
        auc = roc_auc_score(y_test, probs)
    except Exception:
        auc = 0.0

    log.info("Logistic Regression Dormancy | F1=%.3f | P=%.3f | R=%.3f | AUC=%.3f",
             f1, p, r, auc)
    print(classification_report(y_test, preds, target_names=["clean", "dormant_activation"]))

    if use_mlflow:
        try:
            import mlflow, mlflow.sklearn
            mlflow.set_tracking_uri(MLFLOW_URI)
            mlflow.set_experiment("AnomaNet_Classifiers")
            with mlflow.start_run(run_name="LogisticRegression_Dormancy"):
                mlflow.log_params({"C": 1.0, "class_weight": "balanced", "detector": "dormancy"})
                mlflow.log_metrics({"f1": f1, "precision": p, "recall": r, "auc": auc})
                mlflow.sklearn.log_model(model, "logistic_dormancy")
        except Exception as e:
            log.warning("MLflow logging failed: %s", e)

    model._scaler = scaler
    return model


# ── Main ──────────────────────────────────────────────────────────────────────

def main(data_dir: str = "data", use_mlflow: bool = True):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    tx, acct, cust = load_data(data_dir)

    log.info("=" * 55)
    log.info("TRAINING 1/3 — Isolation Forest (Layering)")
    log.info("=" * 55)
    lf = build_layering_features(tx, acct)
    save_model(train_isolation_forest(lf, use_mlflow), "isolation_forest_layering.pkl")

    log.info("=" * 55)
    log.info("TRAINING 2/3 — XGBoost (Structuring)")
    log.info("=" * 55)
    sf = build_structuring_features(tx, acct)
    if len(sf) > 0:
        save_model(train_xgboost(sf, use_mlflow), "xgboost_structuring.pkl")
    else:
        log.warning("No structuring features — skipping XGBoost")

    log.info("=" * 55)
    log.info("TRAINING 3/3 — Logistic Regression (Dormancy)")
    log.info("=" * 55)
    df = build_dormancy_features(tx, acct)
    if len(df) > 0:
        m = train_logistic_regression(df, use_mlflow)
        if m:
            save_model(m, "logistic_dormancy.pkl")
    else:
        log.warning("No dormancy features — skipping LR")

    log.info("=" * 55)
    log.info("ALL MODELS TRAINED — saved to %s", MODELS_DIR.resolve())
    log.info("=" * 55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",      default="data")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()
    main(data_dir=args.data, use_mlflow=not args.no_mlflow)