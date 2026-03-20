"""
ml/core/scoring/layering_scorer.py

Detects rapid layering through multiple accounts.

Dual approach — ensemble of:
  1. Hard velocity rule: deterministic, fires on obvious cases immediately
     (tx_count_1h > 5 AND total_amount_1h > ₹5L → score = 0.80)

  2. Isolation Forest: anomaly scorer trained on velocity features from
     the clean transaction distribution. Catches subtler layering patterns
     that don't hit the hard thresholds.

Final score = max(rule_score, isolation_forest_score)

Detection signals used:
  - tx_count_1h           (fan-out velocity)
  - total_amount_1h       (volume in the window)
  - unique_counterparties_24h (breadth of fan-out)
  - cross_branch_ratio    (fraction of txns across different branches)
  - residency_seconds     (time between inbound and outbound on same account)
  - is_off_hours          (2–5 AM activity)
  - out_degree_1h         (how many distinct accounts received money)
"""

from __future__ import annotations

import logging
import os
import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import numpy as np
from sklearn.ensemble import IsolationForest

log = logging.getLogger(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "isolation_forest_layering.pkl")

# ── Hard rule thresholds ──────────────────────────────────────────────────────
HARD_TX_COUNT_1H       = 5          # more than 5 outbound in 1 hour
HARD_AMOUNT_1H         = 5_00_000   # ₹5 lakhs total in 1 hour
HARD_RULE_SCORE        = 0.80

# ── Feature vector definition (order matters — must match training) ───────────
FEATURE_NAMES = [
    "tx_count_1h",
    "tx_count_24h",
    "total_amount_1h",
    "total_amount_24h",
    "unique_counterparties_24h",
    "out_degree_1h",
    "cross_branch_ratio",
    "residency_seconds",
    "is_off_hours",
    "amount_percentile_in_history",
]


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class LayeringResult:
    account_id:          str
    layering_score:      float
    rule_fired:          bool
    isolation_score:     float
    features:            dict
    explanation_tokens:  dict


# ── Model loading ─────────────────────────────────────────────────────────────

_model: Optional[IsolationForest] = None


def _load_model() -> Optional[IsolationForest]:
    global _model
    if _model is not None:
        return _model
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, "rb") as f:
                _model = pickle.load(f)
            log.info("Isolation Forest loaded from %s", MODEL_PATH)
        except Exception as e:
            log.warning("Failed to load Isolation Forest: %s — using rule only", e)
            _model = None
    else:
        log.warning("Isolation Forest model not found at %s — using rule only", MODEL_PATH)
    return _model


# ── Feature builder ───────────────────────────────────────────────────────────

def _build_feature_vector(features: dict) -> np.ndarray:
    """
    Convert a feature dict into a numpy array in the correct order.
    Missing features default to 0 — safe fallback.
    """
    return np.array(
        [float(features.get(name, 0.0)) for name in FEATURE_NAMES],
        dtype=np.float32,
    ).reshape(1, -1)


def _is_off_hours(ts: Optional[datetime]) -> bool:
    if ts is None:
        return False
    hour = ts.hour
    return 2 <= hour <= 5


# ── Hard rule ─────────────────────────────────────────────────────────────────

def _apply_hard_rule(features: dict) -> float:
    """
    Returns HARD_RULE_SCORE if the deterministic velocity rule fires.
    Returns 0.0 otherwise.

    Rule: tx_count_1h > 5 AND total_amount_1h > ₹5L
    Both conditions must be true — prevents false positives on high-value
    single transfers.
    """
    tx_count_1h    = features.get("tx_count_1h", 0)
    total_amount_1h = features.get("total_amount_1h", 0.0)

    if tx_count_1h > HARD_TX_COUNT_1H and total_amount_1h > HARD_AMOUNT_1H:
        log.debug(
            "Hard velocity rule fired: tx_count_1h=%d, total_amount_1h=%.0f",
            tx_count_1h, total_amount_1h,
        )
        return HARD_RULE_SCORE
    return 0.0


# ── Isolation Forest scoring ──────────────────────────────────────────────────

def _apply_isolation_forest(features: dict) -> float:
    """
    Returns anomaly score from Isolation Forest (0–1).
    Isolation Forest outputs -1 (anomaly) or 1 (normal) from decision_function.
    We normalise to [0, 1]: higher = more anomalous.

    If model not loaded, returns 0.0 (graceful degradation to rule-only mode).
    """
    model = _load_model()
    if model is None:
        return 0.0

    try:
        X = _build_feature_vector(features)
        # decision_function: more negative = more anomalous
        raw_score = model.decision_function(X)[0]
        threshold = getattr(model, '_custom_threshold', 0.5)
        normalised = float(np.clip(0.5 - raw_score, 0.0, 1.0))
        # Only return a meaningful score if anomaly score exceeds threshold
        return normalised if (-raw_score) >= threshold else normalised * 0.3
        
    except Exception as e:
        log.error("Isolation Forest inference failed: %s", e)
        return 0.0


# ── Score adjustments ─────────────────────────────────────────────────────────

def _apply_bonuses(base_score: float, features: dict) -> float:
    """
    Apply contextual bonuses on top of the base score.
    These reward stronger signals without replacing the base detectors.
    """
    score = base_score

    # Off-hours activity is a strong contextual signal
    if features.get("is_off_hours", 0):
        score += 0.05

    # High cross-branch ratio (layering moves money across branches)
    cross_branch = features.get("cross_branch_ratio", 0.0)
    if cross_branch > 0.7:
        score += 0.06
    elif cross_branch > 0.4:
        score += 0.03

    # Very short residency = money arrives and immediately leaves
    residency = features.get("residency_seconds", 9999)
    if 0 < residency < 300:       # under 5 minutes
        score += 0.07
    elif 0 < residency < 900:     # under 15 minutes
        score += 0.04

    # High unique counterparty count in 24h
    unique_cp = features.get("unique_counterparties_24h", 0)
    if unique_cp > 10:
        score += 0.05
    elif unique_cp > 5:
        score += 0.02

    return min(score, 1.0)


# ── Public interface ──────────────────────────────────────────────────────────

def score_layering(
    account_id: str,
    rolling_features: dict,
    current_tx_timestamp: Optional[datetime] = None,
    residency_seconds: float = 9999.0,
) -> LayeringResult:
    """
    Main entry point. Called by anoma_score.py.

    Args:
        account_id: account being scored
        rolling_features: dict from Rupali's get_rolling_features(account_id)
                          Expected keys: tx_count_1h, tx_count_24h,
                          total_amount_24h, unique_counterparties_24h,
                          avg_tx_amount_30d, channel_entropy, cross_branch_ratio
        current_tx_timestamp: datetime of the transaction being scored
        residency_seconds: seconds between inbound and outbound on this account
                           (computed by Kafka consumer from Neo4j)

    Returns:
        LayeringResult with composite score and all feature values.
    """
    # Merge rolling features with transaction-level features
    features = dict(rolling_features)
    features["residency_seconds"]  = residency_seconds
    features["is_off_hours"]       = int(_is_off_hours(current_tx_timestamp))

    # total_amount_1h may not be in rolling features — derive from context
    if "total_amount_1h" not in features:
        features["total_amount_1h"] = features.get("total_amount_24h", 0.0) / 24.0

    # out_degree_1h ≈ tx_count_1h (approximation if not separately tracked)
    if "out_degree_1h" not in features:
        features["out_degree_1h"] = features.get("tx_count_1h", 0)

    # amount_percentile_in_history — default 0.5 if not provided
    if "amount_percentile_in_history" not in features:
        features["amount_percentile_in_history"] = 0.5

    # ── Score ─────────────────────────────────────────────────────────────────
    rule_score  = _apply_hard_rule(features)
    if_score    = _apply_isolation_forest(features)
    base_score  = max(rule_score, if_score)
    final_score = _apply_bonuses(base_score, features)

    result = LayeringResult(
        account_id      = account_id,
        layering_score  = round(final_score, 4),
        rule_fired      = rule_score > 0,
        isolation_score = round(if_score, 4),
        features        = features,
        explanation_tokens = {
            "account_id":           account_id,
            "tx_count_1h":          features.get("tx_count_1h", 0),
            "total_amount_1h":      features.get("total_amount_1h", 0.0),
            "unique_counterparties": features.get("unique_counterparties_24h", 0),
            "cross_branch_ratio":   features.get("cross_branch_ratio", 0.0),
            "residency_seconds":    residency_seconds,
            "is_off_hours":         bool(features.get("is_off_hours", 0)),
            "rule_fired":           rule_score > 0,
        },
    )

    if final_score > 0.5:
        log.info(
            "LAYERING signal | account=%s | score=%.3f | rule=%s | IF=%.3f | "
            "tx_1h=%d | amount_1h=%.0f | residency=%.0fs",
            account_id, final_score, rule_score > 0, if_score,
            features.get("tx_count_1h", 0),
            features.get("total_amount_1h", 0.0),
            residency_seconds,
        )

    return result


# ── Offline scoring for training / evaluation ─────────────────────────────────

def score_layering_from_features(features: dict, account_id: str = "unknown") -> LayeringResult:
    """
    Score directly from a feature dict. Used by training scripts and tests.
    No Redis or Neo4j connection needed.
    """
    return score_layering(
        account_id=account_id,
        rolling_features=features,
        current_tx_timestamp=None,
        residency_seconds=features.get("residency_seconds", 9999.0),
    )