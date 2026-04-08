"""
Affective Knowledge Graph (AKG) — Transition Validator
=======================================================

This module provides deterministic, symbolic validation of emotional sequences
against the sparse OCC-grounded transition matrix defined in
``akg/transition_matrix.py``.  No probabilistic inference, language model, or
external database is involved: validity is computed by direct lookup against the
constraint graph.

The validator operates exclusively over ``EMOTION_SET`` as defined in
``akg/emotion_schema.py``.  Any token not present in ``EMOTION_SET`` is
treated as a hard constraint violation.  There is no fallback, no fuzzy
matching, and no silent remapping: unknown emotions are flagged explicitly in
the return value and counted as invalid transitions.

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
1. **Type check** — reject non-list inputs and non-string elements immediately.
2. **Structural check** — reject empty sequences.
3. **Membership check** — flag any token not in ``EMOTION_SET`` as an unknown
   emotion; unknown emotions generate invalid transition records for every pair
   they participate in.  No fallback or remapping is applied.
4. **Transition check** — for each consecutive pair ``(e_i, e_{i+1})`` where
   both tokens are known, perform direct lookup in ``TRANSITIONS``.
5. **Scoring** — compute ETVS from aggregated counts.

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

    # "surprise" is no longer a valid emotion in EMOTION_SET:
    result = validate_sequence(["fear", "surprise"])
    # {
    #     "valid_transitions": 0,
    #     "invalid_transitions": 1,
    #     "invalid_pairs": [("fear", "surprise")],
    #     "unknown_emotions": ["surprise"],
    #     "etvs": 0.0
    # }
"""

from __future__ import annotations

from akg.emotion_schema import EMOTION_SET
from akg.transition_matrix import TRANSITIONS

# Frozen set for O(1) membership tests throughout this module.
# Built from EMOTION_SET — the single source of truth for the valid
# emotion space.  "surprise" is not a member of EMOTION_SET and will
# therefore be flagged as unknown by all validation calls.
_KNOWN_EMOTIONS: frozenset[str] = frozenset(EMOTION_SET)

# Compile-time guard: assert schema and matrix are consistent.
assert "surprise" not in _KNOWN_EMOTIONS, (
    "'surprise' must not be a member of the validated emotion set. "
    "Check akg/emotion_schema.py."
)


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
        ``EMOTION_SET``; unknown tokens — including ``"surprise"`` — return an
        empty list rather than raising, to support graceful degradation in
        pipeline contexts.

    Returns
    -------
    list[str]
        Sorted list of emotion strings reachable from *emotion* in one step.
        Empty if *emotion* is not a member of ``EMOTION_SET`` or has no
        outgoing transitions.

    Examples
    --------
    ::

        >>> get_allowed_next("fear")
        ['anger', 'distress', 'hope']

        >>> get_allowed_next("nonexistent")
        []

        >>> get_allowed_next("surprise")   # not in EMOTION_SET
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
    * A token absent from ``EMOTION_SET`` — including ``"surprise"`` — is an
      **unknown emotion**.  Every transition pair involving an unknown token is
      recorded as invalid and included in ``invalid_pairs``.  No fallback or
      remapping is applied.  Processing continues for the remainder of the
      sequence so the full violation set is captured in a single pass.
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
        of ``EMOTION_SET``.

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

        >>> validate_sequence(["fear", "surprise"])
        {
            'valid_transitions': 0,
            'invalid_transitions': 1,
            'invalid_pairs': [('fear', 'surprise')],
            'unknown_emotions': ['surprise'],
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
    # Identify unknown tokens
    # Tokens not in EMOTION_SET are strict violations — no fallback applied.
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
        # Any unknown token in the pair makes the transition strictly invalid.
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