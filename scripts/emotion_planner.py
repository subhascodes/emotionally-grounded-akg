"""
scripts/emotion_planner.py
===========================

Deterministic, reproducible emotional trajectory planner for the Affective
Knowledge Graph (AKG) storytelling research prototype.

Theoretical framing
--------------------
**State-space interpretation**
    The AKG transition graph is treated as a finite deterministic state space
    :math:`\\mathcal{G} = (\\mathcal{E}, \\mathcal{T})`, where
    :math:`\\mathcal{E}` is the set of nine OCC emotion nodes and
    :math:`\\mathcal{T} \\subseteq \\mathcal{E} \\times \\mathcal{E}` is the
    sparse set of directed, psychologically licensed transitions.  Each
    emotion is a discrete state; each ``TRANSITIONS_TO`` edge is a legal
    one-step state change.

**Planning formulation**
    Trajectory planning is formalised as a *constrained random walk* of fixed
    horizon :math:`k` on :math:`\\mathcal{G}`.  At each step :math:`t`, the
    planner selects the next state :math:`e_{t+1}` uniformly at random from
    the *filtered* successor set:

    .. math::

        \\text{candidates}(e_t) =
            \\{e' \\mid (e_t, e') \\in \\mathcal{T}\\}
            \\setminus \\{e_{t-1}\\}

    The subtraction of :math:`e_{t-1}` is the **anti-backtrack filter**,
    which prevents immediate oscillations (A → B → A) without forbidding
    later revisits to a previously visited node.  This models a minimal
    narrative coherence constraint: a character's emotional state should not
    immediately reverse within a single story beat.

**Closed-world constraint assumption**
    Any transition pair not explicitly present in ``TRANSITIONS`` is treated
    as *prohibited*, not merely unmodelled.  The planner will never select a
    successor that is absent from the out-edges of the current state.  If the
    filtered candidate set is empty at any step (e.g., the current node has
    only one outgoing edge and that edge points back to the previous node),
    the planner raises a ``DeadEndError`` with a descriptive message, allowing
    the caller to retry with a different seed or start emotion.

**Reproducibility**
    All random choices use a ``random.Random`` instance seeded with the
    caller-supplied ``seed`` parameter.  A ``None`` seed produces
    non-deterministic behaviour suitable for exploratory sampling; an integer
    seed guarantees identical trajectories across runs for a given
    ``(start_emotion, k, seed)`` triple.

Dependencies
------------
* ``akg/emotion_schema.py`` (``EMOTION_LIST``)
* ``akg/transition_matrix.py`` (``TRANSITIONS``)
* Python standard library only — no Neo4j, no LLM, no external packages.

References
----------
Ortony, A., Clore, G. L., & Collins, A. (1988). *The Cognitive Structure of
Emotions*. Cambridge University Press.
Russell, S. J., & Norvig, P. (2020). *Artificial Intelligence: A Modern
Approach* (4th ed.). Pearson. [State-space search formalism]
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Optional

# Ensure project root is on sys.path when script is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from akg.emotion_schema import EMOTION_LIST
from akg.transition_matrix import TRANSITIONS

# Frozen set for O(1) membership tests.
_KNOWN_EMOTIONS: frozenset[str] = frozenset(EMOTION_LIST)


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class DeadEndError(RuntimeError):
    """Raised when the planner reaches a state with no valid forward moves.

    This occurs when the anti-backtrack filter eliminates all outgoing edges
    from the current node — i.e., the node has exactly one outgoing edge and
    that edge points back to the immediately preceding node.

    Attributes
    ----------
    current_emotion:
        The emotion state at which planning became stuck.
    previous_emotion:
        The immediately preceding emotion state whose exclusion caused the
        dead end, or ``None`` if the dead end occurs at step 1.
    step:
        The zero-indexed step number at which the dead end was detected.
    """

    def __init__(
        self,
        current_emotion: str,
        previous_emotion: Optional[str],
        step: int,
    ) -> None:
        self.current_emotion = current_emotion
        self.previous_emotion = previous_emotion
        self.step = step
        super().__init__(
            f"Planning dead end at step {step}: no valid successor from "
            f"{current_emotion!r} (previous state: {previous_emotion!r} is "
            f"excluded by anti-backtrack filter, and no other outgoing edges "
            f"exist). Try a different seed or start_emotion."
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_all_possible_next(emotion: str) -> list[str]:
    """Return all emotions reachable from *emotion* in one step.

    Performs a direct lookup in the in-memory ``TRANSITIONS`` graph.  The
    result reflects the closed-world constraint: only explicitly defined
    outgoing edges are returned.  The anti-backtrack filter is **not** applied
    here; this function returns the raw successor set and is suitable for
    graph inspection, reachability analysis, and UI display.

    Parameters
    ----------
    emotion:
        A string token representing an OCC emotion.  If the token is not a
        member of ``EMOTION_LIST``, an empty list is returned rather than
        raising, to support graceful degradation in pipeline contexts.

    Returns
    -------
    list[str]
        Sorted list of emotion strings reachable from *emotion* in one
        transition step.  Sorting ensures deterministic output independent of
        dictionary insertion order.  Empty if *emotion* is unknown or is a
        graph sink (no outgoing edges in the current schema).

    Examples
    --------
    ::

        >>> get_all_possible_next("fear")
        ['anger', 'distress', 'hope']

        >>> get_all_possible_next("pride")
        ['joy', 'shame']

        >>> get_all_possible_next("nonexistent")
        []
    """
    if emotion not in _KNOWN_EMOTIONS:
        return []
    return sorted(TRANSITIONS.get(emotion, {}).keys())


def plan_emotion_trajectory(
    start_emotion: str,
    k: int,
    seed: Optional[int] = None,
) -> list[str]:
    """Generate a valid k-step emotional trajectory from *start_emotion*.

    Performs a constrained random walk of length *k* on the AKG transition
    graph, beginning at *start_emotion*.  At each step the next emotion is
    chosen uniformly at random from the licensed outgoing edges of the current
    state, minus the immediately preceding state (anti-backtrack filter).

    Parameters
    ----------
    start_emotion:
        The initial emotion state.  Must be a member of ``EMOTION_LIST``.
    k:
        Total number of steps in the trajectory, including the start node.
        A value of ``1`` returns ``[start_emotion]`` with no transitions.
        Must be ≥ 1.
    seed:
        Integer seed for the internal ``random.Random`` instance.  Passing
        the same seed with the same ``(start_emotion, k)`` always produces
        the same trajectory (deterministic mode).  ``None`` (default) uses
        an unpredictable seed (exploratory mode).

    Returns
    -------
    list[str]
        An ordered list of *k* emotion strings.  The first element is
        ``start_emotion``.  Every consecutive pair ``(trajectory[i],
        trajectory[i+1])`` is a valid edge in ``TRANSITIONS``.  No immediate
        A → B → A reversal is present.

    Raises
    ------
    ValueError
        If *start_emotion* is not in ``EMOTION_LIST``, or if *k* < 1.
    DeadEndError
        If the anti-backtrack filter eliminates all candidates at any step.
        The caller may catch this and retry with a different ``seed``.

    Notes
    -----
    The planner does **not** guarantee that all *k*-step trajectories are
    equally likely, only that each step's choice is uniform over the filtered
    candidate set at that step.  Global trajectory distribution depends on
    the graph topology and the anti-backtrack constraint.

    Examples
    --------
    ::

        # Deterministic 5-step trajectory
        >>> plan_emotion_trajectory("distress", k=5, seed=42)
        ['distress', 'anger', 'gratitude', 'joy', 'pride']

        # Single-node trajectory (no transitions required)
        >>> plan_emotion_trajectory("joy", k=1)
        ['joy']

        # Reproducibility: same seed → same result
        >>> t1 = plan_emotion_trajectory("hope", k=4, seed=7)
        >>> t2 = plan_emotion_trajectory("hope", k=4, seed=7)
        >>> assert t1 == t2
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if not isinstance(start_emotion, str):
        raise ValueError(
            f"start_emotion must be a string, got {type(start_emotion).__name__!r}."
        )
    if start_emotion not in _KNOWN_EMOTIONS:
        raise ValueError(
            f"Unknown start_emotion: {start_emotion!r}. "
            f"Valid emotions are: {sorted(_KNOWN_EMOTIONS)}."
        )
    if not isinstance(k, int) or isinstance(k, bool):
        raise ValueError(f"k must be an integer, got {type(k).__name__!r}.")
    if k < 1:
        raise ValueError(f"k must be ≥ 1, got {k}.")

    # ------------------------------------------------------------------
    # Degenerate case
    # ------------------------------------------------------------------
    if k == 1:
        return [start_emotion]

    # ------------------------------------------------------------------
    # Initialise RNG and trajectory buffer
    # ------------------------------------------------------------------
    rng: random.Random = random.Random(seed)
    trajectory: list[str] = [start_emotion]
    previous: Optional[str] = None  # no predecessor at step 0
    current: str = start_emotion

    # ------------------------------------------------------------------
    # Walk k-1 steps
    # ------------------------------------------------------------------
    for step in range(1, k):
        # Raw successors from the transition graph.
        raw_successors: list[str] = get_all_possible_next(current)

        # Apply anti-backtrack filter: exclude the immediately preceding node.
        candidates: list[str] = [
            e for e in raw_successors if e != previous
        ]

        if not candidates:
            raise DeadEndError(
                current_emotion=current,
                previous_emotion=previous,
                step=step,
            )

        # Uniform random choice over filtered candidates (sorted for
        # determinism — sort order is stable across Python versions).
        chosen: str = rng.choice(sorted(candidates))

        trajectory.append(chosen)
        previous = current
        current = chosen

    return trajectory


# ---------------------------------------------------------------------------
# CLI demo entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Demonstrate the planner with a range of start emotions and seeds."""
    demo_cases: list[tuple[str, int, Optional[int]]] = [
        ("distress",  6, 42),
        ("hope",      5, 0),
        ("surprise",  7, 99),
        ("fear",      4, 1337),
        ("joy",       1, None),   # degenerate: k=1, no transitions
        ("shame",     5, 7),
    ]

    print("=" * 70)
    print("AKG Emotion Trajectory Planner — Demo Output")
    print("=" * 70)

    for start, k, seed in demo_cases:
        print(f"\nstart={start!r:12s}  k={k}  seed={seed}")
        print("-" * 50)
        try:
            trajectory = plan_emotion_trajectory(start, k=k, seed=seed)
            arrow_path = "  →  ".join(trajectory)
            print(f"Trajectory : {arrow_path}")
            print(f"Length     : {len(trajectory)}")
        except DeadEndError as exc:
            print(f"DeadEndError: {exc}")
        except ValueError as exc:
            print(f"ValueError: {exc}")

    # Show raw successor sets for reference.
    print("\n" + "=" * 70)
    print("Raw successor sets (anti-backtrack filter NOT applied)")
    print("=" * 70)
    for emotion in sorted(_KNOWN_EMOTIONS):
        nexts = get_all_possible_next(emotion)
        print(f"  {emotion:12s} → {nexts}")


if __name__ == "__main__":
    main()