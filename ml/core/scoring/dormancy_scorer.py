"""
ml/core/scoring/dormancy_scorer.py

Detects sudden activation of dormant accounts.

Two-stage approach:
  Stage 1 — State machine rule:
    if account.is_dormant AND current_tx.amount > 10× historical_avg
    → base_score = 0.75

  Stage 2 — Logistic regression adjusts the score based on:
    - dormancy_duration_months  (longer dormancy = higher risk)
    - post_activation_outbound_speed  (how fast money left after arrival)
    - counterparty_risk          (are destinations flagged accounts?)
    - kyc_recently_updated       (flag in transaction metadata)
    - amount_vs_declared_income  (inbound vs declared monthly income)

Final score = logistic_regression(features) if model loaded,
              else state_machine_score with manual bonuses.
"""

from __future__ import annotations

import logging
import os
import pickle
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np

from core.graph.neo4j_client import get_account_features, get_historical_avg_amount

log = logging.getLogger(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "logistic_dormancy.pkl")

# ── State machine thresholds ──────────────────────────────────────────────────
DORMANCY_MONTHS_MIN          = 12     # RBI definition: 12+ months inactive
AMOUNT_MULTIPLIER_THRESHOLD  = 10     # inbound > 10× historical avg = suspicious
BASE_STATE_MACHINE_SCORE     = 0.75

FEATURE_NAMES = [
    "dormancy_duration_months",
    "amount_vs_historical_avg_ratio",
    "amount_vs_declared_income_ratio",
    "post_activation_outbound_hours",
    "kyc_recently_updated",
    "is_high_kyc_risk",
    "inbound_amount_log",
]


@dataclass
class DormancyResult:
    account_id:                  str
    dormancy_score:              float
    is_dormant:                  bool
    dormancy_duration_months:    float
    amount_vs_avg_ratio:         float
    state_machine_fired:         bool
    explanation_tokens:          dict


_NULL = lambda aid: DormancyResult(
    account_id=aid, dormancy_score=0.0, is_dormant=False,
    dormancy_duration_months=0.0, amount_vs_avg_ratio=0.0,
    state_machine_fired=False, explanation_tokens={},
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
            log.info("Logistic regression dormancy model loaded")
        except Exception as e:
            log.warning("Failed to load dormancy model: %s", e)
    return _model


def _months_dormant(dormant_since: Optional[str]) -> float:
    if not dormant_since:
        return 0.0
    try:
        since_dt = datetime.fromisoformat(dormant_since)
        if since_dt.tzinfo is None:
            since_dt = since_dt.replace(tzinfo=timezone.utc)
        delta = datetime.now(tz=timezone.utc) - since_dt
        return delta.days / 30.44
    except Exception:
        return 0.0


def _logistic_score(features: dict) -> float:
    model = _load_model()
    if model is None:
        return 0.0
    try:
        X = np.array(
            [float(features.get(n, 0.0)) for n in FEATURE_NAMES],
            dtype=np.float32,
        ).reshape(1, -1)
        prob = model.predict_proba(X)[0][1]
        return float(prob)
    except Exception as e:
        log.error("Logistic regression inference failed: %s", e)
        return 0.0


def _rule_bonuses(base_score: float, features: dict) -> float:
    score = base_score

    # Long dormancy = higher risk
    months = features.get("dormancy_duration_months", 0.0)
    if months >= 24:
        score += 0.10
    elif months >= 18:
        score += 0.06
    elif months >= 12:
        score += 0.03

    # Money left very quickly after arriving
    outbound_hours = features.get("post_activation_outbound_hours", 999.0)
    if 0 < outbound_hours < 2:
        score += 0.10
    elif 0 < outbound_hours < 6:
        score += 0.06
    elif 0 < outbound_hours < 12:
        score += 0.03

    # KYC was recently updated (classic takeover signal)
    if features.get("kyc_recently_updated", 0):
        score += 0.08

    # High-risk KYC tier
    if features.get("is_high_kyc_risk", 0):
        score += 0.04

    # Extreme amount vs income ratio
    income_ratio = features.get("amount_vs_declared_income_ratio", 0.0)
    if income_ratio > 100:
        score += 0.07
    elif income_ratio > 50:
        score += 0.04

    return min(score, 1.0)


def score_dormancy(
    account_id: str,
    current_amount: float,
    current_tx_metadata: Optional[dict] = None,
    post_activation_outbound_hours: float = 999.0,
) -> DormancyResult:
    """
    Main entry point. Called by anoma_score.py.

    Args:
        account_id: account being scored
        current_amount: amount of the transaction triggering the score
        current_tx_metadata: dict from the transaction (checked for kyc_recently_updated)
        post_activation_outbound_hours: hours between inbound and first outbound
                                        (computed by Kafka consumer)

    Returns:
        DormancyResult with score and dormancy metadata.
    """
    # ── Fetch account state from Neo4j ────────────────────────────────────────
    account_features = get_account_features(account_id)

    if not account_features:
        return _NULL(account_id)

    is_dormant    = account_features.get("is_dormant", False)
    dormant_since = account_features.get("dormant_since")

    # Only apply dormancy scorer to dormant accounts
    if not is_dormant and not dormant_since:
        return _NULL(account_id)

    # ── Historical baseline ───────────────────────────────────────────────────
    hist_avg = get_historical_avg_amount(account_id, days=365)

    # If no history at all, use declared income as proxy (very conservative)
    if hist_avg == 0:
        declared_income = float(account_features.get("declared_monthly_income") or 0)
        hist_avg = declared_income * 0.1 if declared_income > 0 else 1_000.0

    amount_ratio = current_amount / hist_avg if hist_avg > 0 else 0.0

    # ── Stage 1: State machine ────────────────────────────────────────────────
    state_machine_fired = (
        is_dormant and
        amount_ratio >= AMOUNT_MULTIPLIER_THRESHOLD
    )

    if not state_machine_fired:
        # Still score lightly if dormancy period is long, even if ratio < 10×
        months = _months_dormant(dormant_since)
        if months < DORMANCY_MONTHS_MIN:
            return _NULL(account_id)
        # Partial score for accounts that are dormant but transaction isn't huge
        partial = min(0.35 + (months / 36) * 0.2, 0.55)
        return DormancyResult(
            account_id=account_id, dormancy_score=round(partial, 4),
            is_dormant=True, dormancy_duration_months=round(months, 1),
            amount_vs_avg_ratio=round(amount_ratio, 2),
            state_machine_fired=False,
            explanation_tokens={"account_id": account_id, "partial": True},
        )

    months = _months_dormant(dormant_since)
    meta   = current_tx_metadata or {}
    kyc_updated = int(bool(meta.get("kyc_recently_updated", False)))

    declared_income = float(account_features.get("declared_monthly_income") or 0)
    income_ratio    = (current_amount / declared_income) if declared_income > 0 else 0.0

    kyc_tier     = account_features.get("kyc_risk_tier", "LOW")
    is_high_risk = int(kyc_tier in ("HIGH", "PEP"))

    features = {
        "dormancy_duration_months":         round(months, 2),
        "amount_vs_historical_avg_ratio":   round(amount_ratio, 2),
        "amount_vs_declared_income_ratio":  round(income_ratio, 2),
        "post_activation_outbound_hours":   post_activation_outbound_hours,
        "kyc_recently_updated":             kyc_updated,
        "is_high_kyc_risk":                 is_high_risk,
        "inbound_amount_log":               float(np.log1p(current_amount)),
    }

    # ── Stage 2: logistic regression or rule bonuses ──────────────────────────
    lr_score = _logistic_score(features)
    if lr_score > 0:
        final_score = lr_score
    else:
        final_score = _rule_bonuses(BASE_STATE_MACHINE_SCORE, features)

    final_score = round(min(final_score, 1.0), 4)

    log.info(
        "DORMANCY signal | account=%s | score=%.3f | dormant_months=%.1f | "
        "amount_ratio=%.1fx | outbound_hours=%.1f",
        account_id, final_score, months, amount_ratio, post_activation_outbound_hours,
    )

    return DormancyResult(
        account_id               = account_id,
        dormancy_score           = final_score,
        is_dormant               = True,
        dormancy_duration_months = round(months, 1),
        amount_vs_avg_ratio      = round(amount_ratio, 2),
        state_machine_fired      = True,
        explanation_tokens       = {
            "account_id":            account_id,
            "dormancy_months":       round(months, 1),
            "amount_ratio":          round(amount_ratio, 1),
            "current_amount":        round(current_amount, 2),
            "outbound_hours":        post_activation_outbound_hours,
            "kyc_recently_updated":  bool(kyc_updated),
        },
    )