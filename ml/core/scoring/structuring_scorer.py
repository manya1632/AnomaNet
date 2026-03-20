"""
ml/core/scoring/structuring_scorer.py

Detects structuring / smurfing — breaking large cash amounts into
multiple transactions just below regulatory reporting thresholds.

India's CTR thresholds:
  ₹10,00,000 — primary Cash Transaction Report threshold
  ₹5,00,000  — secondary
  ₹2,00,000  — tertiary

Model: XGBoost binary classifier trained on features derived from the
       transaction window around the current transaction.

Features:
  - n_txns_below_threshold_7d   (count of near-threshold txns in 7 days)
  - aggregate_amount_7d         (total cash in the window)
  - n_distinct_branches_7d      (smurfing = multiple branches)
  - min_time_delta_hours        (how close together the transactions are)
  - max_time_delta_hours        (spread of the window)
  - closest_threshold           (which CTR tier is being targeted)
  - avg_pct_below_threshold     (how close to threshold on average)
  - is_cash_channel             (CASH or BRANCH channel)
  - declared_income_ratio       (aggregate vs declared monthly income)
"""

from __future__ import annotations

import logging
import os
import pickle
from dataclasses import dataclass
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "xgboost_structuring.pkl")

# ── CTR threshold tiers ───────────────────────────────────────────────────────
CTR_THRESHOLDS = [10_00_000, 5_00_000, 2_00_000]

# An amount is "near threshold" if it's in [threshold × 0.85, threshold × 0.99]
NEAR_THRESHOLD_LOW  = 0.85
NEAR_THRESHOLD_HIGH = 0.99

FEATURE_NAMES = [
    "n_txns_below_threshold_7d",
    "aggregate_amount_7d",
    "n_distinct_branches_7d",
    "min_time_delta_hours",
    "max_time_delta_hours",
    "closest_threshold",
    "avg_pct_below_threshold",
    "is_cash_channel",
    "declared_income_ratio",
]


@dataclass
class StructuringResult:
    account_id:         str
    structuring_score:  float
    threshold_tier:     Optional[int]   # ₹10L / ₹5L / ₹2L
    n_suspicious_txns:  int
    aggregate_amount:   float
    explanation_tokens: dict


_NULL = lambda aid: StructuringResult(
    account_id=aid, structuring_score=0.0, threshold_tier=None,
    n_suspicious_txns=0, aggregate_amount=0.0, explanation_tokens={},
)

_model = None


def _load_model():
    global _model
    if _model is not None:
        return _model
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, "rb") as f:
                _model = pickle.load(f)
            log.info("XGBoost structuring model loaded")
        except Exception as e:
            log.warning("Failed to load XGBoost model: %s", e)
    else:
        log.warning("XGBoost model not found — using rule-based fallback")
    return _model


def _nearest_threshold(amount: float) -> tuple[int, float]:
    """Return (threshold_value, pct_of_threshold) for the closest CTR tier."""
    best_threshold = CTR_THRESHOLDS[0]
    best_pct       = 0.0
    for t in CTR_THRESHOLDS:
        pct = amount / t
        if NEAR_THRESHOLD_LOW <= pct <= NEAR_THRESHOLD_HIGH:
            if pct > best_pct:
                best_pct       = pct
                best_threshold = t
    return best_threshold, best_pct


def _rule_based_score(features: dict) -> float:
    """
    Deterministic fallback when XGBoost model isn't loaded yet.
    Returns a score based on the strength of structuring signals.
    """
    n_txns = features.get("n_txns_below_threshold_7d", 0)
    if n_txns < 3:
        return 0.0

    score = 0.0

    # Base score from number of suspicious transactions
    if n_txns >= 6:
        score = 0.80
    elif n_txns >= 4:
        score = 0.70
    else:
        score = 0.60

    # Boost if smurfing (multiple branches)
    branches = features.get("n_distinct_branches_7d", 1)
    if branches >= 3:
        score += 0.10
    elif branches >= 2:
        score += 0.05

    # Boost if aggregate is a large multiple of the threshold
    agg   = features.get("aggregate_amount_7d", 0.0)
    tier  = features.get("closest_threshold", 10_00_000)
    if tier > 0 and agg > tier * 2.5:
        score += 0.05

    # Boost if close together in time (rapid structuring)
    min_delta = features.get("min_time_delta_hours", 999.0)
    if min_delta < 2:
        score += 0.05

    return min(score, 1.0)


def _xgboost_score(features: dict) -> float:
    model = _load_model()
    if model is None:
        return 0.0
    try:
        X = np.array(
            [float(features.get(n, 0.0)) for n in FEATURE_NAMES],
            dtype=np.float32,
        ).reshape(1, -1)
        prob = model.predict_proba(X)[0][1]   # probability of fraud class
        return float(prob)
    except Exception as e:
        log.error("XGBoost inference failed: %s", e)
        return 0.0


def score_structuring(
    account_id: str,
    recent_transactions: list[dict],
    declared_monthly_income: float = 0.0,
) -> StructuringResult:
    """
    Main entry point. Called by anoma_score.py.

    Args:
        account_id: account being scored
        recent_transactions: list of transaction dicts for this account,
            last 7 days. Each dict needs: amount, channel, branch_id, initiated_at
        declared_monthly_income: from KYC — used for income ratio feature

    Returns:
        StructuringResult with score and structuring metadata.
    """
    if not recent_transactions:
        return _NULL(account_id)

    # ── Build features ────────────────────────────────────────────────────────
    cash_txns = [
        t for t in recent_transactions
        if t.get("channel") in ("CASH", "BRANCH")
    ]

    # Find transactions near any CTR threshold
    suspicious_txns   = []
    threshold_hits    = {t: [] for t in CTR_THRESHOLDS}

    for tx in cash_txns:
        amt = float(tx.get("amount", 0))
        for threshold in CTR_THRESHOLDS:
            pct = amt / threshold
            if NEAR_THRESHOLD_LOW <= pct <= NEAR_THRESHOLD_HIGH:
                suspicious_txns.append(tx)
                threshold_hits[threshold].append(tx)
                break   # count each transaction once

    n_suspicious = len(suspicious_txns)
    if n_suspicious < 2:
        return _NULL(account_id)

    # Identify the primary threshold being targeted
    targeted_threshold = max(threshold_hits, key=lambda t: len(threshold_hits[t]))
    suspicious_for_tier = threshold_hits[targeted_threshold]

    if len(suspicious_for_tier) < 2:
        return _NULL(account_id)

    # Compute features
    amounts    = [float(t.get("amount", 0)) for t in suspicious_for_tier]
    branches   = list({t.get("branch_id", "") for t in suspicious_for_tier})
    timestamps = []
    for tx in suspicious_for_tier:
        ts = tx.get("initiated_at")
        if isinstance(ts, str):
            try:
                from datetime import datetime
                ts = datetime.fromisoformat(ts)
                timestamps.append(ts)
            except Exception:
                pass
        elif hasattr(ts, "hour"):
            timestamps.append(ts)

    time_deltas_hours = []
    if len(timestamps) >= 2:
        timestamps_sorted = sorted(timestamps)
        for i in range(1, len(timestamps_sorted)):
            delta = (timestamps_sorted[i] - timestamps_sorted[i-1]).total_seconds() / 3600
            time_deltas_hours.append(delta)

    avg_pct = float(np.mean([a / targeted_threshold for a in amounts]))
    agg     = sum(amounts)

    features = {
        "n_txns_below_threshold_7d": n_suspicious,
        "aggregate_amount_7d":       agg,
        "n_distinct_branches_7d":    len(branches),
        "min_time_delta_hours":      min(time_deltas_hours) if time_deltas_hours else 0.0,
        "max_time_delta_hours":      max(time_deltas_hours) if time_deltas_hours else 0.0,
        "closest_threshold":         targeted_threshold,
        "avg_pct_below_threshold":   avg_pct,
        "is_cash_channel":           1.0,
        "declared_income_ratio":     (agg / declared_monthly_income)
                                     if declared_monthly_income > 0 else 0.0,
    }

    # ── Score: XGBoost if available, else rule-based ──────────────────────────
    xgb_score  = _xgboost_score(features)
    rule_score = _rule_based_score(features)
    final      = xgb_score if xgb_score > 0 else rule_score
    final      = round(min(final, 1.0), 4)

    if final > 0.5:
        log.info(
            "STRUCTURING signal | account=%s | score=%.3f | tier=₹%s | "
            "n_txns=%d | agg=%.0f | branches=%d",
            account_id, final,
            f"{targeted_threshold:,}",
            n_suspicious, agg, len(branches),
        )

    return StructuringResult(
        account_id        = account_id,
        structuring_score = final,
        threshold_tier    = targeted_threshold,
        n_suspicious_txns = n_suspicious,
        aggregate_amount  = round(agg, 2),
        explanation_tokens = {
            "account_id":         account_id,
            "n_txns":             n_suspicious,
            "threshold_tier":     targeted_threshold,
            "aggregate_amount":   round(agg, 2),
            "n_branches":         len(branches),
            "avg_pct_threshold":  round(avg_pct * 100, 1),
            "min_delta_hours":    round(min(time_deltas_hours), 2) if time_deltas_hours else 0,
        },
    )