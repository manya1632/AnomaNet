"""
ml/interfaces.py
★ MUSKAN creates this file | RUPALI implements the real functions

This is the ONLY file Muskan imports from Rupali's codebase.

Muskan imports:
    from ml.interfaces import score_profile_mismatch
    from ml.interfaces import get_rolling_features
    from ml.interfaces import get_explanation

How this works:
    This file tries to import Rupali's real implementations.
    If her module is not ready yet, safe fallbacks run instead.
    Muskan's pipeline never crashes waiting for Rupali's code.
    Once Rupali implements her modules, they load automatically —
    zero changes needed anywhere in Muskan's files.

Fallback behaviour:
    score_profile_mismatch  → 0.0   (no contribution to AnomaScore)
    get_rolling_features    → dict of zeros (layering falls back to rule-only)
    get_explanation         → generic explanation string from score breakdown
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)


# ── score_profile_mismatch ────────────────────────────────────────────────────

try:
    from ml.modules.profile_mismatch.inference import score as _real_pm_score
    _PM_REAL = True
    log.info("profile_mismatch module loaded")
except ImportError:
    _real_pm_score = None
    _PM_REAL = False
    log.warning("profile_mismatch not available — returning 0.0 until Rupali's module is ready")


def score_profile_mismatch(account_id: str) -> float:
    """
    Returns 0–1 anomaly score: how much the account's recent transaction
    behaviour deviates from its declared KYC profile.

    Implemented by: Rupali — ml/modules/profile_mismatch/inference.py
    Called by:      Muskan — ml/core/scoring/anoma_score.py
    Fallback:       0.0
    """
    if _PM_REAL and _real_pm_score is not None:
        try:
            return float(_real_pm_score(account_id))
        except Exception as e:
            log.error("score_profile_mismatch(%s) failed: %s", account_id, e)
    return 0.0


# ── get_rolling_features ──────────────────────────────────────────────────────

try:
    from ml.shared.feature_store.redis_store import get_rolling_features as _real_features
    _FEATURES_REAL = True
    log.info("feature_store module loaded")
except ImportError:
    _real_features = None
    _FEATURES_REAL = False
    log.warning("feature_store not available — returning zero features until Rupali's module is ready")

# All keys Muskan's scorers expect — zero values are safe defaults
_ZERO_FEATURES: dict = {
    "tx_count_1h":               0,
    "tx_count_24h":              0,
    "total_amount_24h":          0.0,
    "unique_counterparties_24h": 0,
    "avg_tx_amount_30d":         0.0,
    "channel_entropy":           0.0,
    "cross_branch_ratio":        0.0,
}


def get_rolling_features(account_id: str) -> dict:
    """
    Returns rolling-window velocity features for the account from Redis.

    Keys: tx_count_1h, tx_count_24h, total_amount_24h,
          unique_counterparties_24h, avg_tx_amount_30d,
          channel_entropy, cross_branch_ratio

    Implemented by: Rupali — ml/shared/feature_store/redis_store.py
    Called by:      Muskan — ml/core/scoring/anoma_score.py
    Fallback:       dict of zeros (layering scorer uses rule-only mode)
    """
    if _FEATURES_REAL and _real_features is not None:
        try:
            result = _real_features(account_id)
            # Merge with zero-dict to guarantee all keys present
            return {**_ZERO_FEATURES, **result}
        except Exception as e:
            log.error("get_rolling_features(%s) failed: %s", account_id, e)
    return dict(_ZERO_FEATURES)


# ── get_explanation ───────────────────────────────────────────────────────────

try:
    from ml.modules.explainability.generator import get_explanation as _real_explanation
    _EXPLAIN_REAL = True
    log.info("explainability module loaded")
except ImportError:
    _real_explanation = None
    _EXPLAIN_REAL = False
    log.warning("explainability not available — returning generic explanation until Rupali's module is ready")


def get_explanation(alert_id: str, score_breakdown: dict) -> str:
    """
    Returns a plain-English explanation paragraph for why the alert fired.

    Args:
        alert_id:        UUID of the alert
        score_breakdown: {layering, circular, structuring, dormancy,
                          profile_mismatch} — each a float 0–1

    Implemented by: Rupali — ml/modules/explainability/generator.py
    Called by:      Muskan — ml/core/main.py POST /ml/explain
    Fallback:       generic explanation from score breakdown
    """
    if _EXPLAIN_REAL and _real_explanation is not None:
        try:
            return _real_explanation(alert_id, score_breakdown)
        except Exception as e:
            log.error("get_explanation(%s) failed: %s", alert_id, e)

    return _generic_explanation(alert_id, score_breakdown)


def _generic_explanation(alert_id: str, score_breakdown: dict) -> str:
    if not score_breakdown:
        return (
            f"Alert {alert_id} was flagged by AnomaNet's fraud detection pipeline. "
            "Detailed explanation will be available once all modules are loaded."
        )

    dominant       = max(score_breakdown, key=lambda k: score_breakdown.get(k, 0))
    dominant_score = score_breakdown.get(dominant, 0)

    descriptions = {
        "circular":         "circular fund flow (round-tripping)",
        "layering":         "rapid layering through multiple accounts",
        "structuring":      "structuring of cash deposits below CTR thresholds",
        "dormancy":         "sudden activation of a dormant account",
        "profile_mismatch": "transaction behaviour inconsistent with declared KYC profile",
    }

    other_patterns = [
        descriptions.get(k, k)
        for k, v in score_breakdown.items()
        if k != dominant and v > 0.4
    ]

    explanation = (
        f"This account was flagged primarily for {descriptions.get(dominant, dominant)} "
        f"(score: {dominant_score:.2f}). "
    )
    if other_patterns:
        explanation += f"Additional signals detected: {' and '.join(other_patterns)}. "

    explanation += (
        "The composite AnomaScore exceeded the alert threshold. "
        "Full narrative explanation will be generated by the explainability engine."
    )
    return explanation