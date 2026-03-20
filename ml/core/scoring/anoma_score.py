"""
ml/core/scoring/anoma_score.py

Composite AnomaScore aggregator.

Formula (runtime-configurable weights, defaults from blueprint):
  AnomaScore = 0.25×layering + 0.30×circular + 0.20×structuring
             + 0.10×dormancy + 0.15×profile_mismatch

Thresholds:
  Standard accounts : 0.65
  PEP-tier accounts : 0.45

This module calls all five detectors and packages results into
AnomaScoreResult — the single object published by the Kafka consumer.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from core.scoring.circular_detector  import score_circular,    CycleResult
from core.scoring.layering_scorer    import score_layering,    LayeringResult
from core.scoring.structuring_scorer import score_structuring, StructuringResult
from core.scoring.dormancy_scorer    import score_dormancy,    DormancyResult

# Rupali's interface — called via interfaces.py
from ml.interfaces import score_profile_mismatch, get_rolling_features

log = logging.getLogger(__name__)

# ── Default weights (match blueprint exactly) ─────────────────────────────────
DEFAULT_WEIGHTS = {
    "layering":        0.25,
    "circular":        0.30,
    "structuring":     0.20,
    "dormancy":        0.10,
    "profile_mismatch": 0.15,
}

DEFAULT_THRESHOLD     = float(os.getenv("ANOMASCORE_THRESHOLD", "0.65"))
PEP_THRESHOLD         = float(os.getenv("PEP_THRESHOLD",        "0.45"))


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class AnomaScoreResult:
    transaction_id:     str
    account_id:         str
    anoma_score:        float          # 0.0 – 1.0 composite
    threshold_used:     float
    alert_triggered:    bool

    # Per-detector scores
    score_breakdown: dict = field(default_factory=dict)
    # {layering: f, circular: f, structuring: f, dormancy: f, profile_mismatch: f}

    detected_patterns: list[str] = field(default_factory=list)
    # e.g. ["CIRCULAR", "LAYERING"]

    # Rich metadata from each detector (for explainability engine)
    circular_result:     Optional[CycleResult]      = None
    layering_result:     Optional[LayeringResult]   = None
    structuring_result:  Optional[StructuringResult] = None
    dormancy_result:     Optional[DormancyResult]   = None
    profile_mismatch_score: float                   = 0.0

    scored_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


# ── Weight configuration (updated by admin panel at runtime) ──────────────────

_current_weights = dict(DEFAULT_WEIGHTS)


def update_weights(new_weights: dict):
    """
    Update scoring weights at runtime. Called by admin settings panel via
    PUT /api/admin/config → Ratnesh's backend → POST /ml/score config update.
    Weights must sum to 1.0.
    """
    total = sum(new_weights.values())
    if abs(total - 1.0) > 0.01:
        raise ValueError(f"Weights must sum to 1.0, got {total:.3f}")
    _current_weights.update(new_weights)
    log.info("AnomaScore weights updated: %s", _current_weights)


def get_weights() -> dict:
    return dict(_current_weights)


# ── Threshold logic ───────────────────────────────────────────────────────────

def _get_threshold(kyc_risk_tier: str) -> float:
    return PEP_THRESHOLD if kyc_risk_tier == "PEP" else DEFAULT_THRESHOLD


# ── Main scoring pipeline ─────────────────────────────────────────────────────

def compute_anoma_score(
    transaction_id: str,
    account_id: str,
    current_amount: float,
    current_channel: str,
    current_tx_timestamp: datetime,
    current_tx_metadata: Optional[dict] = None,
    recent_transactions: Optional[list[dict]] = None,
    kyc_risk_tier: str = "LOW",
    declared_monthly_income: float = 0.0,
    post_activation_outbound_hours: float = 999.0,
    residency_seconds: float = 9999.0,
) -> AnomaScoreResult:
    """
    Run all five detectors and compute the composite AnomaScore.

    Args:
        transaction_id:   UUID of the transaction being scored
        account_id:       primary account (source or destination)
        current_amount:   transaction amount in INR
        current_channel:  NEFT / RTGS / UPI / SWIFT etc.
        current_tx_timestamp: datetime of the transaction
        current_tx_metadata:  raw metadata dict from the transaction
        recent_transactions:  list of recent tx dicts (last 7 days) for structuring
        kyc_risk_tier:    LOW / MEDIUM / HIGH / PEP
        declared_monthly_income: from KYC record
        post_activation_outbound_hours: seconds from inbound to outbound
        residency_seconds: time between in and out on this account

    Returns:
        AnomaScoreResult with composite score and all detector outputs.
    """
    meta = current_tx_metadata or {}
    weights = _current_weights

    # ── 1. Get rolling features from Rupali's Redis store ────────────────────
    try:
        rolling = get_rolling_features(account_id)
    except Exception as e:
        log.warning("get_rolling_features failed for %s: %s — using empty dict", account_id, e)
        rolling = {}

    # ── 2. Circular detector ─────────────────────────────────────────────────
    try:
        circular_result = score_circular(account_id, hours=72)
    except Exception as e:
        log.error("Circular detector failed: %s", e)
        from core.scoring.circular_detector import _NULL_RESULT
        circular_result = _NULL_RESULT(account_id)

    # ── 3. Layering scorer ───────────────────────────────────────────────────
    try:
        layering_result = score_layering(
            account_id=account_id,
            rolling_features=rolling,
            current_tx_timestamp=current_tx_timestamp,
            residency_seconds=residency_seconds,
        )
    except Exception as e:
        log.error("Layering scorer failed: %s", e)
        from core.scoring.layering_scorer import LayeringResult
        layering_result = LayeringResult(
            account_id=account_id, layering_score=0.0, rule_fired=False,
            isolation_score=0.0, features={}, explanation_tokens={},
        )

    # ── 4. Structuring scorer ────────────────────────────────────────────────
    try:
        structuring_result = score_structuring(
            account_id=account_id,
            recent_transactions=recent_transactions or [],
            declared_monthly_income=declared_monthly_income,
        )
    except Exception as e:
        log.error("Structuring scorer failed: %s", e)
        from core.scoring.structuring_scorer import _NULL as S_NULL
        structuring_result = S_NULL(account_id)

    # ── 5. Dormancy scorer ───────────────────────────────────────────────────
    try:
        dormancy_result = score_dormancy(
            account_id=account_id,
            current_amount=current_amount,
            current_tx_metadata=meta,
            post_activation_outbound_hours=post_activation_outbound_hours,
        )
    except Exception as e:
        log.error("Dormancy scorer failed: %s", e)
        from core.scoring.dormancy_scorer import _NULL as D_NULL
        dormancy_result = D_NULL(account_id)

    # ── 6. Profile mismatch (Rupali's module via interfaces.py) ─────────────
    try:
        pm_score = score_profile_mismatch(account_id)
    except Exception as e:
        log.warning("score_profile_mismatch failed for %s: %s", account_id, e)
        pm_score = 0.0

    # ── 7. Composite AnomaScore ──────────────────────────────────────────────
    score_breakdown = {
        "layering":         round(layering_result.layering_score, 4),
        "circular":         round(circular_result.cycle_score, 4),
        "structuring":      round(structuring_result.structuring_score, 4),
        "dormancy":         round(dormancy_result.dormancy_score, 4),
        "profile_mismatch": round(pm_score, 4),
    }

    anoma_score = round(
        weights["layering"]         * score_breakdown["layering"]        +
        weights["circular"]         * score_breakdown["circular"]        +
        weights["structuring"]      * score_breakdown["structuring"]     +
        weights["dormancy"]         * score_breakdown["dormancy"]        +
        weights["profile_mismatch"] * score_breakdown["profile_mismatch"],
        4,
    )

    # ── 8. Detected patterns ─────────────────────────────────────────────────
    PATTERN_THRESHOLD = 0.50   # a pattern is "detected" if its score > 0.5
    detected = []
    if score_breakdown["layering"]         > PATTERN_THRESHOLD: detected.append("LAYERING")
    if score_breakdown["circular"]         > PATTERN_THRESHOLD: detected.append("CIRCULAR")
    if score_breakdown["structuring"]      > PATTERN_THRESHOLD: detected.append("STRUCTURING")
    if score_breakdown["dormancy"]         > PATTERN_THRESHOLD: detected.append("DORMANT")
    if score_breakdown["profile_mismatch"] > PATTERN_THRESHOLD: detected.append("PROFILE_MISMATCH")

    threshold      = _get_threshold(kyc_risk_tier)
    alert_triggered = anoma_score >= threshold

    result = AnomaScoreResult(
        transaction_id      = transaction_id,
        account_id          = account_id,
        anoma_score         = anoma_score,
        threshold_used      = threshold,
        alert_triggered     = alert_triggered,
        score_breakdown     = score_breakdown,
        detected_patterns   = detected,
        circular_result     = circular_result,
        layering_result     = layering_result,
        structuring_result  = structuring_result,
        dormancy_result     = dormancy_result,
        profile_mismatch_score = pm_score,
    )

    log.info(
        "AnomaScore | tx=%s | account=%s | score=%.3f | threshold=%.2f | "
        "alert=%s | patterns=%s",
        transaction_id, account_id, anoma_score, threshold,
        alert_triggered, detected or "none",
    )

    return result