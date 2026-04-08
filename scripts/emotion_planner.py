"""
scripts/emotion_planner.py
==========================
Deterministic, graph-constrained emotion trajectory planner for the
neuro-symbolic storytelling system.

All transitions are resolved exclusively by querying the Neo4j Affective
Knowledge Graph (AKG).  No static transition data is used; the planner
cannot function without a live Neo4j connection.

Graph schema
------------
Nodes : (:Emotion {name: str})
Edges : (:Emotion)-[:TRANSITION]->(:Emotion)
Valid emotions: joy | distress | hope | fear | pride | shame | anger | gratitude

Design properties
-----------------
* Deterministic  — identical (start_emotion, k, seed) always yields the same
                   trajectory regardless of call order or graph query timing.
* Graph-faithful — every step in the trajectory corresponds to a real edge
                   stored in Neo4j; no transition is inferred or hardcoded.
* Fail-fast      — a missing outgoing edge raises RuntimeError immediately;
                   silent fallbacks are explicitly prohibited.
* Traceable      — optional DEBUG_PLANNER=1 prints every chosen transition,
                   supporting reproducibility claims in the accompanying paper.
"""

import os
import random
from typing import Optional

from akg.neo4j_connector import get_neighbors
from akg.emotion_schema import EMOTION_SET, is_valid_emotion

# ── Debug configuration ───────────────────────────────────────────────────────
_DEBUG: bool = os.getenv("DEBUG_PLANNER", "0") == "1"


# ── Internal helpers ──────────────────────────────────────────────────────────

def _debug(current: str, chosen: str) -> None:
    """
    Emit a single transition trace line when debug mode is active.

    Output format (for paper traceability):
        [PLANNER] distress → shame
    """
    if _DEBUG:
        print(f"[PLANNER] {current} → {chosen}")


def _resolve_next(
    current: str,
    previous: Optional[str],
) -> list[str]:
    """
    Fetch valid next emotions from Neo4j and apply the no-backtrack filter.

    Steps
    -----
    1. Query Neo4j for all outgoing neighbors of *current*.
    2. Raise RuntimeError immediately if the list is empty — no fallback.
    3. Remove *previous* from candidates to suppress immediate back-tracking.
    4. Restore the full neighbor list if filtering empties it (graph topology
       corner-case: only one outgoing edge that points back to previous).

    Args:
        current  (str):           The emotion at the current trajectory position.
        previous (str | None):    The emotion at the previous position, or None
                                  on the first step.

    Returns:
        list[str]: Non-empty candidate list for random.choice.

    Raises:
        RuntimeError: If Neo4j returns no outgoing edges for *current*.
    """
    neighbors: list[str] = get_neighbors(current)

    if not neighbors:
        raise RuntimeError(
            f"[PLANNER] Dead end: no outgoing TRANSITION edges found for "
            f"emotion '{current}' in the AKG.  Ensure the graph is fully "
            f"populated before running the planner."
        )

    # Suppress immediate back-tracking
    filtered: list[str] = (
        [n for n in neighbors if n != previous]
        if previous is not None
        else neighbors
    )

    # Fallback: single-exit node whose only neighbor is the previous emotion
    if not filtered:
        filtered = neighbors

    return filtered


# ── Public API ────────────────────────────────────────────────────────────────

def get_next_candidates(current_emotion: str) -> list[str]:
    """
    Return a sorted list of valid next emotions from the Neo4j AKG.

    This helper is intentionally stateless and seed-independent; it is
    provided for inspection and unit-testing purposes.

    Args:
        current_emotion (str): A valid OCC emotion label.

    Returns:
        list[str]: Alphabetically sorted list of reachable emotions.

    Raises:
        ValueError:   If *current_emotion* is not in EMOTION_SET.
        RuntimeError: If Neo4j returns no outgoing edges.
    """
    if not is_valid_emotion(current_emotion):
        raise ValueError(
            f"'{current_emotion}' is not a recognised OCC emotion.  "
            f"Valid set: {sorted(EMOTION_SET)}."
        )

    neighbors: list[str] = get_neighbors(current_emotion)

    if not neighbors:
        raise RuntimeError(
            f"No outgoing transitions found for emotion: '{current_emotion}'."
        )

    return sorted(neighbors)


def plan_emotion_trajectory(
    start_emotion: str,
    k: int,
    seed: int = 42,
) -> list[str]:
    """
    Generate a deterministic, graph-constrained emotional trajectory.

    The function performs a seeded random walk over the Neo4j AKG.  At each
    step the set of valid successors is retrieved from the graph; the next
    emotion is chosen via seeded random.choice, guaranteeing reproducibility.

    Parameters
    ----------
    start_emotion : str
        Seed emotion for the trajectory.  Must be a member of EMOTION_SET and
        must have at least one outgoing TRANSITION edge in the AKG.
    k : int
        Total trajectory length, **including** the start emotion.  Must be ≥ 2.
    seed : int, optional
        Random seed.  Identical (start_emotion, k, seed) triples always produce
        identical trajectories (default: 42).

    Returns
    -------
    list[str]
        Ordered trajectory [e₀, e₁, …, e_{k-1}] of length *k*.
        Every consecutive pair (eᵢ, eᵢ₊₁) corresponds to a real
        TRANSITION edge in the AKG.

    Raises
    ------
    ValueError
        If *start_emotion* is not in EMOTION_SET, or *k* < 2.
    RuntimeError
        If any emotion encountered during the walk has no outgoing edges in
        the AKG (dead end).  The planner does not silently recover.
    """
    # ── Validate inputs ───────────────────────────────────────────────────────
    if not is_valid_emotion(start_emotion):
        raise ValueError(
            f"start_emotion '{start_emotion}' is not a recognised OCC emotion.  "
            f"Valid set: {sorted(EMOTION_SET)}."
        )
    if k < 2:
        raise ValueError(
            f"Trajectory length k must be ≥ 2; received k={k}."
        )

    # ── Initialise seeded PRNG and trajectory ─────────────────────────────────
    random.seed(seed)
    trajectory: list[str] = [start_emotion]

    # ── Graph walk ────────────────────────────────────────────────────────────
    for _ in range(k - 1):
        current:  str           = trajectory[-1]
        previous: Optional[str] = trajectory[-2] if len(trajectory) >= 2 else None

        candidates: list[str] = _resolve_next(current, previous)
        chosen:     str        = random.choice(candidates)

        _debug(current, chosen)
        trajectory.append(chosen)

    return trajectory


# ── CLI test block ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from akg.neo4j_connector import close_driver

    START = "distress"
    K     = 4
    SEED  = 42

    print("=" * 60)
    print("EMOTION PLANNER — Neo4j AKG Trajectory Test")
    print("=" * 60)
    print(f"  start_emotion : {START}")
    print(f"  k             : {K}")
    print(f"  seed          : {SEED}")
    print()

    try:
        # ── Primary trajectory ────────────────────────────────────────────────
        trajectory = plan_emotion_trajectory(
            start_emotion=START,
            k=K,
            seed=SEED,
        )
        print("Planned trajectory:")
        print("  " + " → ".join(trajectory))
        print()

        # ── Determinism check ─────────────────────────────────────────────────
        print("Determinism check (same seed → identical output):")
        all_same = True
        for trial in range(1, 4):
            t = plan_emotion_trajectory(start_emotion=START, k=K, seed=SEED)
            match = "✓" if t == trajectory else "✗"
            if t != trajectory:
                all_same = False
            print(f"  Trial {trial} {match}: {' → '.join(t)}")
        print(f"  Result: {'PASS — all identical' if all_same else 'FAIL — mismatch detected'}")
        print()

        # ── Variation check ───────────────────────────────────────────────────
        print("Variation check (different seeds → different trajectories):")
        seen: set = {tuple(trajectory)}
        for s in [0, 1, 7, 13, 99]:
            t = plan_emotion_trajectory(start_emotion=START, k=K, seed=s)
            tag = "(same)" if tuple(t) in seen else "(different)"
            seen.add(tuple(t))
            print(f"  seed={s:>3}: {' → '.join(t)}  {tag}")

    except (ValueError, RuntimeError) as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    finally:
        close_driver()

    print()
    print("=" * 60)