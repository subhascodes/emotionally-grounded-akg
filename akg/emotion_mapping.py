"""
akg/emotion_mapping.py
=======================

Deterministic utility layer for mapping arbitrary emotion labels onto the
canonical OCC emotion set defined in ``akg/emotion_schema.py``.

This module serves as the single point of truth for any component in the
neuro-symbolic storytelling pipeline that must normalise an externally
produced emotion string (e.g., from a third-party classifier, a dataset
annotation, or a free-text label) into one of the nine AKG-valid emotions.

Design principles
-----------------
**Strict schema alignment**
    The only valid output values are members of ``EMOTION_LIST`` as imported
    from ``akg/emotion_schema.py``.  No additional emotions are introduced or
    permitted.

**Determinism**
    ``map_to_occ`` is a pure function with no randomness, no external I/O,
    and no mutable state.  Given the same input string it always returns the
    same output string.

**Priority-ordered keyword matching**
    Each input label is lower-cased and checked against an ordered list of
    ``(keyword, occ_emotion)`` pairs.  The first keyword found as a
    case-insensitive substring of the input wins.  Priority order is
    documented in ``_KEYWORD_PRIORITY`` and follows the rule specific → general
    to prevent dominant emotional states from being overridden by weaker or
    more transient appraisals.

**Safe fallback**
    If no keyword matches, ``map_to_occ`` returns ``"distress"`` — the most
    semantically neutral negative-valence OCC emotion — rather than raising.
    This prevents downstream graph traversal from failing on unrecognised
    labels while flagging the ambiguity to callers via the predictable return
    value.

**Validation utility**
    ``is_valid_emotion`` provides an O(1) membership test against the frozen
    schema set, suitable for use as an assertion guard or pipeline filter.

References
----------
Ortony, A., Clore, G. L., & Collins, A. (1988). *The Cognitive Structure of
Emotions*. Cambridge University Press.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path when this module is used directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from akg.emotion_schema import EMOTION_LIST

# ---------------------------------------------------------------------------
# Schema anchor
# ---------------------------------------------------------------------------

# Frozen set for O(1) membership tests in is_valid_emotion().
# Derived exclusively from the imported EMOTION_LIST; never extended here.
_VALID_EMOTIONS: frozenset[str] = frozenset(EMOTION_LIST)

# ---------------------------------------------------------------------------
# Keyword priority table
# ---------------------------------------------------------------------------

# Each entry is a (keyword, occ_emotion) pair.  The list is evaluated in
# order; the FIRST matching keyword wins.  Keywords are matched as
# case-insensitive substrings of the lowercased input label.
#
# Priority rationale (specific → general):
#   - Shame and anger are placed first because their keywords are highly
#     specific and rarely ambiguous; allowing distress or fear to match
#     before them would produce incorrect OCC appraisal-branch assignments.
#   - Fear follows shame and anger; its keywords (scared, anxious, nervous,
#     worried, panic) are specific enough not to bleed into other categories.
#   - Distress is placed after the above agent-directed and threat-based
#     emotions to act as the general negative-valence bucket.
#   - Positive-valence emotions (joy, hope, gratitude, pride) follow.
#   - "surprise" is placed last because in OCC theory it is a transient
#     modifier emotion triggered by expectation violation, not a dominant
#     sustained emotional state.  Placing it last prevents transient surprise
#     cues from overriding substantive appraisals (e.g., "astonished at the
#     betrayal" should map to a valenced emotion, not surprise).

_KEYWORD_PRIORITY: list[tuple[str, str]] = [
    # ------------------------------------------------------------------
    # shame  — agent-based, self-directed, negative; highly specific keywords
    # ------------------------------------------------------------------
    ("ashamed",      "shame"),
    ("guilty",       "shame"),
    ("embarrassed",  "shame"),

    # ------------------------------------------------------------------
    # anger  — agent-based, other-directed, negative; specific action cues
    # ------------------------------------------------------------------
    ("furious",      "anger"),
    ("annoyed",      "anger"),
    ("angry",        "anger"),
    ("frustrated",   "anger"),

    # ------------------------------------------------------------------
    # fear  — event-based, prospective, negative; threat-detection cues
    # ------------------------------------------------------------------
    ("panic",        "fear"),
    ("scared",       "fear"),
    ("anxious",      "fear"),
    ("nervous",      "fear"),
    ("worried",      "fear"),

    # ------------------------------------------------------------------
    # distress  — event-based, realised, negative; general loss/sadness cues
    # ------------------------------------------------------------------
    ("depressed",    "distress"),
    ("disappointed", "distress"),
    ("grief",        "distress"),
    ("hurt",         "distress"),
    ("sad",          "distress"),
    ("upset",        "distress"),

    # ------------------------------------------------------------------
    # joy  — event-based, realised, positive
    # ------------------------------------------------------------------
    ("delighted",    "joy"),
    ("excited",      "joy"),   # excited maps to joy, not surprise
    ("happy",        "joy"),
    ("pleased",      "joy"),

    # ------------------------------------------------------------------
    # hope  — event-based, prospective, positive
    # ------------------------------------------------------------------
    ("looking forward", "hope"),   # multi-word phrase checked first
    ("hopeful",         "hope"),
    ("optimistic",      "hope"),

    # ------------------------------------------------------------------
    # gratitude  — agent-based, other-directed, positive
    # ------------------------------------------------------------------
    ("appreciative", "gratitude"),
    ("grateful",     "gratitude"),
    ("thankful",     "gratitude"),

    # ------------------------------------------------------------------
    # pride  — agent-based, self-directed, positive
    # ------------------------------------------------------------------
    ("accomplished", "pride"),
    ("confident",    "pride"),
    ("proud",        "pride"),

    # ------------------------------------------------------------------
    # surprise  — LOWEST PRIORITY: valence-neutral, transient OCC modifier.
    # In OCC theory, surprise reflects expectation violation rather than a
    # sustained appraisal state.  It is placed last so that substantive
    # valenced emotions (e.g., distress, joy) take precedence when their
    # keywords co-occur with surprise cues in a compound label.
    # ------------------------------------------------------------------
    ("astonished",   "surprise"),
    ("shocked",      "surprise"),
    ("amazed",       "surprise"),
]

# Compile-time assertion: every target in the priority table must be a
# member of the AKG schema.  This guard fires at import time so a schema
# change is caught immediately without requiring a test run.
for _kw, _target in _KEYWORD_PRIORITY:
    assert _target in _VALID_EMOTIONS, (
        f"Keyword table references unknown OCC emotion {_target!r}. "
        f"Valid emotions: {sorted(_VALID_EMOTIONS)}"
    )

# Safe fallback returned when no keyword matches.
_FALLBACK_EMOTION: str = "distress"
assert _FALLBACK_EMOTION in _VALID_EMOTIONS, (
    f"Fallback emotion {_FALLBACK_EMOTION!r} is not in EMOTION_LIST."
)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def map_to_occ(label: str) -> str:
    """Map an arbitrary emotion label onto the nearest AKG OCC emotion.

    Performs a deterministic, priority-ordered keyword search over the
    lowercased input string.  The first keyword found as a substring of the
    input determines the return value.  If no keyword matches, the safe
    fallback ``"distress"`` is returned.

    The function never raises on unexpected input: non-string arguments are
    coerced to strings via ``str()`` before processing.

    Parameters
    ----------
    label:
        An arbitrary emotion label string.  May be a single word (e.g.,
        ``"angry"``), a compound phrase (e.g., ``"very sad and upset"``), or
        an OCC emotion name already in the schema.  Case-insensitive.

    Returns
    -------
    str
        A member of ``EMOTION_LIST``.  Always one of:
        ``"joy"``, ``"distress"``, ``"hope"``, ``"fear"``,
        ``"pride"``, ``"shame"``, ``"anger"``, ``"gratitude"``,
        ``"surprise"``.

    Notes
    -----
    * If the lowercased input is already a valid OCC emotion, it is returned
      immediately without consulting the keyword table, preserving exact
      schema labels at zero cost.
    * Keyword matching uses substring containment (``keyword in label``),
      so a keyword matches even if embedded in a longer phrase.  The first
      match in priority order wins; see ``_KEYWORD_PRIORITY`` for the defined
      order and its rationale.
    * Multi-word keywords (e.g., ``"looking forward"``) are evaluated before
      their component words to prevent partial matches from shadowing them.
    * ``"excited"`` maps to ``"joy"`` (not ``"surprise"``): excitement is an
      anticipatory positive-valence state in OCC terms, not an expectation-
      violation response.

    Examples
    --------
    ::

        >>> map_to_occ("angry")
        'anger'

        >>> map_to_occ("GUILTY")
        'shame'

        >>> map_to_occ("very sad and upset")
        'distress'

        >>> map_to_occ("joy")                # already a valid OCC label
        'joy'

        >>> map_to_occ("confused")           # no keyword match → fallback
        'distress'

        >>> map_to_occ("")                   # empty string → fallback
        'distress'

        >>> map_to_occ("excited")            # joy, not surprise
        'joy'

        >>> map_to_occ("looking forward")    # multi-word hope cue
        'hope'

        >>> map_to_occ("shocked and ashamed")  # shame beats surprise
        'shame'
    """
    normalised: str = str(label).lower().strip()

    # Fast path: input is already a canonical OCC emotion.
    if normalised in _VALID_EMOTIONS:
        return normalised

    # Priority-ordered keyword scan.
    for keyword, occ_emotion in _KEYWORD_PRIORITY:
        if keyword in normalised:
            return occ_emotion

    # No match: return safe fallback.
    return _FALLBACK_EMOTION


def is_valid_emotion(label: str) -> bool:
    """Return ``True`` if *label* is a member of the AKG OCC emotion schema.

    Performs an exact, case-sensitive membership test against ``EMOTION_LIST``.
    Intended as a lightweight guard for pipeline components that must verify
    an emotion string before passing it to the planner or validator.

    Parameters
    ----------
    label:
        The emotion string to test.

    Returns
    -------
    bool
        ``True`` if *label* is exactly one of the nine OCC emotions in
        ``EMOTION_LIST``; ``False`` otherwise.

    Examples
    --------
    ::

        >>> is_valid_emotion("anger")
        True

        >>> is_valid_emotion("Anger")   # case-sensitive: capital A fails
        False

        >>> is_valid_emotion("rage")    # not in schema
        False

        >>> is_valid_emotion("")
        False
    """
    return label in _VALID_EMOTIONS