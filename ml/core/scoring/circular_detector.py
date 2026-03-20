"""
ml/core/scoring/circular_detector.py

Detects circular / round-tripping fraud patterns.

Two-stage approach:
  Stage 1 — Neo4j pre-filter: Cypher finds candidate cycles starting and
             ending at the account within the time window. Fast, uses the
             graph index. Returns raw paths with amounts and timestamps.

  Stage 2 — NetworkX validation: Johnson's Algorithm on the extracted
             subgraph confirms cycles and applies the fraud criteria:
               • cycle length 2–7 hops
               • completed within 72 hours
               • edge amounts within ±15% variance across the cycle
               • ≥2 first-time counterparty relationships

Output: CycleResult dataclass with score (0–1), detected flag, path, and
        all metadata needed by the explainability engine.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

import networkx as nx
import numpy as np

from core.graph.neo4j_client import (
    get_subgraph,
    get_cycle_candidates,
    is_first_time_counterparty,
)

log = logging.getLogger(__name__)

# ── Fraud thresholds ──────────────────────────────────────────────────────────
MAX_CYCLE_HOURS        = 72      # cycle must complete within 72 hours
MIN_CYCLE_LENGTH       = 2       # minimum hops in a valid cycle
MAX_CYCLE_LENGTH       = 7       # maximum hops
AMOUNT_VARIANCE_LIMIT  = 0.15    # edge amounts within ±15% of cycle mean
MIN_FIRST_TIME_EDGES   = 2       # at least 2 first-time counterparty pairs


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class CycleResult:
    account_id:         str
    cycle_detected:     bool
    cycle_score:        float           # 0.0 – 1.0
    cycle_path:         list[str]       # account IDs in cycle order
    amounts:            list[float]     # one per edge in cycle
    completion_hours:   float           # time from first to last edge
    amount_variance:    float           # coefficient of variation of amounts
    first_time_edges:   int             # count of new counterparty pairs
    explanation_tokens: dict = field(default_factory=dict)


_NULL_RESULT = lambda aid: CycleResult(
    account_id=aid, cycle_detected=False, cycle_score=0.0,
    cycle_path=[], amounts=[], completion_hours=0.0,
    amount_variance=0.0, first_time_edges=0,
)


# ── Core detection logic ──────────────────────────────────────────────────────

def _amount_variance(amounts: list[float]) -> float:
    """Coefficient of variation: std/mean. 0 = perfectly equal amounts."""
    if not amounts or len(amounts) < 2:
        return 0.0
    arr  = np.array(amounts, dtype=float)
    mean = arr.mean()
    if mean == 0:
        return 0.0
    return float(arr.std() / mean)


def _cycle_duration_hours(timestamps: list[datetime]) -> float:
    """Hours between first and last transaction in a cycle."""
    valid = [t for t in timestamps if t is not None]
    if len(valid) < 2:
        return 0.0
    return (max(valid) - min(valid)).total_seconds() / 3600


def _validate_cycle(
    path: list[str],
    amounts: list[float],
    timestamps: list[datetime],
) -> tuple[bool, float, dict]:
    """
    Apply all fraud criteria to a candidate cycle.
    Returns (is_valid, score, metadata_dict).

    Scoring logic:
      Base score 0.70 if all hard criteria pass.
      Bonus +0.10 if completed in < 6 hours (very rapid cycle)
      Bonus +0.08 if amount variance < 0.05 (suspiciously uniform)
      Bonus +0.07 if ≥3 first-time edges (all new relationships)
      Penalty -0.15 if variance > AMOUNT_VARIANCE_LIMIT (amounts differ a lot)
    """
    hop_count = len(path) - 1   # path includes start node twice for closed cycle

    # ── Hard criteria ─────────────────────────────────────────────────────────
    if not (MIN_CYCLE_LENGTH <= hop_count <= MAX_CYCLE_LENGTH):
        return False, 0.0, {}

    variance = _amount_variance(amounts)
    if variance > AMOUNT_VARIANCE_LIMIT:
        return False, 0.0, {}

    duration_hours = _cycle_duration_hours(timestamps)
    if duration_hours > MAX_CYCLE_HOURS:
        return False, 0.0, {}

    # Count first-time counterparty relationships
    first_time_count = 0
    for i in range(len(path) - 1):
        if is_first_time_counterparty(path[i], path[i + 1]):
            first_time_count += 1

    if first_time_count < MIN_FIRST_TIME_EDGES:
        return False, 0.0, {}

    # ── Scoring ───────────────────────────────────────────────────────────────
    score = 0.70

    if duration_hours < 6:
        score += 0.10
    elif duration_hours < 24:
        score += 0.05

    if variance < 0.05:
        score += 0.08
    elif variance < 0.10:
        score += 0.04

    if first_time_count >= 3:
        score += 0.07
    elif first_time_count >= 2:
        score += 0.03

    score = min(score, 1.0)

    meta = {
        "hop_count":       hop_count,
        "duration_hours":  round(duration_hours, 2),
        "amount_variance": round(variance, 4),
        "first_time_edges": first_time_count,
        "amounts":         [round(a, 2) for a in amounts],
    }

    return True, score, meta


def _detect_via_networkx(
    G: nx.DiGraph,
    account_id: str,
) -> list[tuple[list[str], float, dict]]:
    """
    Run Johnson's Algorithm on the extracted subgraph.
    Returns a list of (path, score, metadata) for every valid cycle
    that passes all fraud criteria.

    Johnson's finds ALL simple directed cycles. We then filter by
    our fraud criteria. This is the definitive detection step.
    """
    valid_cycles = []

    try:
        all_cycles = list(nx.simple_cycles(G))
    except Exception as e:
        log.error("Johnson's Algorithm failed: %s", e)
        return []

    for cycle in all_cycles:
        # Only consider cycles that include our target account
        if account_id not in cycle:
            continue

        hop_count = len(cycle)
        if not (MIN_CYCLE_LENGTH <= hop_count <= MAX_CYCLE_LENGTH):
            continue

        # Rotate cycle so account_id is first
        idx   = cycle.index(account_id)
        cycle = cycle[idx:] + cycle[:idx]

        # Build closed path: [A, B, C, A]
        closed_path = cycle + [cycle[0]]

        # Extract amounts and timestamps from graph edges
        amounts    = []
        timestamps = []
        valid      = True

        for i in range(len(cycle)):
            src = cycle[i]
            dst = cycle[(i + 1) % len(cycle)]
            if G.has_edge(src, dst):
                edge_data = G[src][dst]
                amounts.append(edge_data.get("amount", 0.0))
                ts = edge_data.get("timestamp")
                if isinstance(ts, datetime):
                    timestamps.append(ts)
            else:
                valid = False
                break

        if not valid or not amounts:
            continue

        is_valid, score, meta = _validate_cycle(closed_path, amounts, timestamps)
        if is_valid:
            valid_cycles.append((closed_path, score, meta))

    # Return the highest-scoring cycle if multiple found
    valid_cycles.sort(key=lambda x: x[1], reverse=True)
    return valid_cycles


# ── Public interface ──────────────────────────────────────────────────────────

def score_circular(
    account_id: str,
    hours: int = 72,
    use_neo4j_prefilter: bool = True,
) -> CycleResult:
    """
    Main entry point. Called by anoma_score.py.

    Args:
        account_id: the account being scored
        hours: look-back window for transactions
        use_neo4j_prefilter: if True, use Cypher to get candidate cycles
                             before running Johnson's. Faster for production.
                             Set False for testing with small in-memory graphs.

    Returns:
        CycleResult with score and full cycle metadata.
    """
    # ── Stage 1: get subgraph from Neo4j ─────────────────────────────────────
    G = get_subgraph(account_id, hops=MAX_CYCLE_LENGTH, hours=hours)

    if G.number_of_edges() == 0:
        return _NULL_RESULT(account_id)

    # ── Stage 2: run Johnson's Algorithm ─────────────────────────────────────
    valid_cycles = _detect_via_networkx(G, account_id)

    if not valid_cycles:
        return _NULL_RESULT(account_id)

    # Take the best cycle
    best_path, best_score, best_meta = valid_cycles[0]

    result = CycleResult(
        account_id       = account_id,
        cycle_detected   = True,
        cycle_score      = round(best_score, 4),
        cycle_path       = best_path,
        amounts          = best_meta.get("amounts", []),
        completion_hours = best_meta.get("duration_hours", 0.0),
        amount_variance  = best_meta.get("amount_variance", 0.0),
        first_time_edges = best_meta.get("first_time_edges", 0),
        explanation_tokens = {
            "account_id":       account_id,
            "path":             best_path,
            "amounts":          best_meta.get("amounts", []),
            "duration_hours":   best_meta.get("duration_hours", 0.0),
            "hop_count":        best_meta.get("hop_count", 0),
            "first_time_edges": best_meta.get("first_time_edges", 0),
            "n_cycles_found":   len(valid_cycles),
        },
    )

    log.info(
        "CIRCULAR detected | account=%s | score=%.3f | hops=%d | hours=%.1fh | path=%s",
        account_id, best_score,
        best_meta.get("hop_count", 0),
        best_meta.get("duration_hours", 0.0),
        " → ".join(best_path),
    )

    return result


# ── Offline scoring (used during training / evaluation) ──────────────────────

def score_circular_from_graph(
    G: nx.DiGraph,
    account_id: str,
) -> CycleResult:
    """
    Score circular patterns from a pre-built NetworkX graph.
    Used by training scripts and unit tests — no Neo4j connection needed.
    """
    if G.number_of_edges() == 0:
        return _NULL_RESULT(account_id)

    valid_cycles = _detect_via_networkx(G, account_id)

    if not valid_cycles:
        return _NULL_RESULT(account_id)

    best_path, best_score, best_meta = valid_cycles[0]

    return CycleResult(
        account_id       = account_id,
        cycle_detected   = True,
        cycle_score      = round(best_score, 4),
        cycle_path       = best_path,
        amounts          = best_meta.get("amounts", []),
        completion_hours = best_meta.get("duration_hours", 0.0),
        amount_variance  = best_meta.get("amount_variance", 0.0),
        first_time_edges = best_meta.get("first_time_edges", 0),
        explanation_tokens = {
            "account_id":     account_id,
            "path":           best_path,
            "amounts":        best_meta.get("amounts", []),
            "duration_hours": best_meta.get("duration_hours", 0.0),
            "hop_count":      best_meta.get("hop_count", 0),
        },
    )