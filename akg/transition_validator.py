"""
Affective Knowledge Graph (AKG) — Transition Validator
=======================================================

This module provides deterministic, symbolic validation of emotional sequences
against the sparse OCC-grounded transition matrix defined in
``akg/transition_matrix.py``.  No probabilistic inference, language model, or
external database is involved: validity is computed by direct lookup against the
constraint graph.

Theoretical basis
-----------------
The validator operationalises the closed-world assumption over the AKG
transition graph: any emotion pair ``(e_i, e_{i+1})`` that does not appear as
an explicit directed edge in ``TRANSITIONS`` is treated as a constraint
violation, regardless of intuitive plausibility.  This matches the design
intent of the matrix — absent edges are *constrained out*, not merely
unmodelled.

The **Emotional Transition Validity Score (ETVS)** provides a normalised
scalar summary of sequence conformance:

.. math::

    \\text{ETVS} = \\frac{|\\text{valid transitions}|}{|\\text{total transitions}|}

where ``|total transitions| = len(sequence) - 1``.  A sequence of length 1
(zero transitions) is defined to have ETVS = 1.0, as no constraint can be
violated.  A sequence of length 0 is rejected as malformed before scoring.

Validation pipeline
-------------------
1. **Structural check** — reject empty sequences immediately.
2. **Membership check** — flag any token not in ``EMOTION_LIST`` as an unknown
   emotion; unknown emotions generate invalid transition records for every pair
   they participate in.
3. **Transition check** — for each consecutive pair ``(e_i, e_{i+1})`` where
   both tokens are known, perform direct lookup in ``TRANSITIONS``.
4. **Scoring** — compute ETVS from aggregated counts.

Usage example
-------------
::

    from akg.transition_validator import validate_sequence, get_allowed_next

    result = validate_sequence(["fear", "anger", "shame", "distress"])
    print(result)
    # {
    #     "valid_transitions": 3,
    #     "invalid_transitions": 0,
    #     "invalid_pairs": [],
    #     "unknown_emotions": [],
    #     "etvs": 1.0
    # }

    print(get_allowed_next("shame"))
    # ["anger", "distress", "pride"]
"""

from __future__ import annotations

from akg.emotion_schema import EMOTION_LIST
from akg.transition_matrix import TRANSITIONS

# Frozen set for O(1) membership tests throughout this module.
_KNOWN_EMOTIONS: frozenset[str] = frozenset(EMOTION_LIST)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_allowed_next(emotion: str) -> list[str]:
    """Return the list of emotions that may immediately follow *emotion*.

    Performs a direct lookup in the AKG transition graph.  The result reflects
    the closed-world constraint set: only explicitly defined outgoing edges are
    returned.

    Parameters
    ----------
    emotion:
        A string token representing an OCC emotion.  Must be a member of
        ``EMOTION_LIST``; unknown tokens return an empty list rather than
        raising, to support graceful degradation in pipeline contexts.

    Returns
    -------
    list[str]
        Sorted list of emotion strings reachable from *emotion* in one step.
        Empty if *emotion* is unknown or has no outgoing transitions (a sink
        node in the graph, though none exist in the current schema).

    Examples
    --------
    ::

        >>> get_allowed_next("fear")
        ['anger', 'distress', 'hope']

        >>> get_allowed_next("nonexistent")
        []
    """
    if emotion not in _KNOWN_EMOTIONS:
        return []
    return sorted(TRANSITIONS.get(emotion, {}).keys())


def validate_sequence(sequence: list[str]) -> dict:
    """Validate an emotional sequence against the AKG transition constraints.

    Evaluates every consecutive pair ``(sequence[i], sequence[i+1])`` against
    the directed edges in ``TRANSITIONS`` and computes the Emotional Transition
    Validity Score (ETVS).

    Validation semantics
    --------------------
    * An **empty sequence** raises ``ValueError`` — a sequence must contain at
      least one emotion token to be semantically interpretable.
    * A **single-token sequence** is valid by definition (ETVS = 1.0; zero
      transitions means zero violations).
    * A token absent from ``EMOTION_LIST`` is an **unknown emotion**.  Every
      transition pair involving an unknown token is recorded as invalid and
      included in ``invalid_pairs``, but processing continues for the remainder
      of the sequence.
    * A transition ``(e_i, e_{i+1})`` where both tokens are known but the pair
      is absent from ``TRANSITIONS[e_i]`` is an **invalid transition**.

    Parameters
    ----------
    sequence:
        Ordered list of emotion string tokens representing a narrative
        emotional progression, e.g. ``["hope", "fear", "anger"]``.

    Returns
    -------
    dict with the following keys:

    ``valid_transitions`` : int
        Count of consecutive pairs that satisfy a defined graph edge.

    ``invalid_transitions`` : int
        Count of consecutive pairs that violate the constraint graph, including
        pairs involving unknown emotion tokens.

    ``invalid_pairs`` : list[tuple[str, str]]
        Ordered list of ``(source, target)`` pairs that failed validation.
        Preserves sequence order; duplicates are retained if the same illegal
        transition recurs.

    ``unknown_emotions`` : list[str]
        Deduplicated, sorted list of tokens in *sequence* that are not members
        of ``EMOTION_LIST``.

    ``etvs`` : float
        Emotional Transition Validity Score in ``[0.0, 1.0]``.
        Defined as ``valid_transitions / total_transitions``.
        Returns ``1.0`` when ``total_transitions == 0``.

    Raises
    ------
    ValueError
        If *sequence* is empty.
    TypeError
        If *sequence* is not a list, or contains non-string elements.

    Examples
    --------
    ::

        >>> validate_sequence(["hope", "joy", "pride"])
        {
            'valid_transitions': 2,
            'invalid_transitions': 0,
            'invalid_pairs': [],
            'unknown_emotions': [],
            'etvs': 1.0
        }

        >>> validate_sequence(["joy", "shame"])
        {
            'valid_transitions': 0,
            'invalid_transitions': 1,
            'invalid_pairs': [('joy', 'shame')],
            'unknown_emotions': [],
            'etvs': 0.0
        }

        >>> validate_sequence(["euphoria", "joy"])
        {
            'valid_transitions': 0,
            'invalid_transitions': 1,
            'invalid_pairs': [('euphoria', 'joy')],
            'unknown_emotions': ['euphoria'],
            'etvs': 0.0
        }
    """
    # ------------------------------------------------------------------
    # Type guard
    # ------------------------------------------------------------------
    if not isinstance(sequence, list):
        raise TypeError(
            f"sequence must be a list, got {type(sequence).__name__!r}"
        )
    for i, token in enumerate(sequence):
        if not isinstance(token, str):
            raise TypeError(
                f"All sequence elements must be strings; "
                f"element at index {i} is {type(token).__name__!r}"
            )

    # ------------------------------------------------------------------
    # Structural guard
    # ------------------------------------------------------------------
    if len(sequence) == 0:
        raise ValueError(
            "sequence must contain at least one emotion token; "
            "empty sequences are not interpretable as narrative progressions."
        )

    # ------------------------------------------------------------------
    # Identify unknown tokens (deduplicated, order-preserving insertion)
    # ------------------------------------------------------------------
    unknown_emotions: list[str] = sorted(
        {token for token in sequence if token not in _KNOWN_EMOTIONS}
    )

    # ------------------------------------------------------------------
    # Degenerate case: single token, zero transitions
    # ------------------------------------------------------------------
    if len(sequence) == 1:
        return {
            "valid_transitions": 0,
            "invalid_transitions": 0,
            "invalid_pairs": [],
            "unknown_emotions": unknown_emotions,
            "etvs": 1.0,
        }

    # ------------------------------------------------------------------
    # Evaluate consecutive pairs
    # ------------------------------------------------------------------
    valid_count: int = 0
    invalid_count: int = 0
    invalid_pairs: list[tuple[str, str]] = []

    for src, tgt in zip(sequence, sequence[1:]):
        # Any unknown token in the pair makes the transition invalid.
        if src not in _KNOWN_EMOTIONS or tgt not in _KNOWN_EMOTIONS:
            invalid_count += 1
            invalid_pairs.append((src, tgt))
            continue

        # Both tokens are known: check directed edge existence.
        if tgt in TRANSITIONS.get(src, {}):
            valid_count += 1
        else:
            invalid_count += 1
            invalid_pairs.append((src, tgt))

    # ------------------------------------------------------------------
    # Compute ETVS
    # ------------------------------------------------------------------
    total_transitions: int = valid_count + invalid_count
    # Guard is technically redundant for len >= 2, but kept for explicitness.
    etvs: float = (
        valid_count / total_transitions if total_transitions > 0 else 1.0
    )

    return {
        "valid_transitions": valid_count,
        "invalid_transitions": invalid_count,
        "invalid_pairs": invalid_pairs,
        "unknown_emotions": unknown_emotions,
        "etvs": round(etvs, 6),
    }