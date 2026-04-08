"""
akg/emotion_mapping.py
=======================

Deterministic utility layer for mapping classifier output labels onto the
canonical OCC emotion set defined in ``akg/emotion_schema.py``.

This module is the single point of truth for label normalisation across the
neuro-symbolic storytelling pipeline.  It is designed for use with the
``j-hartmann/emotion-english-distilroberta-base`` classifier, whose output
labels are mapped directly and exhaustively to the eight-emotion
``EMOTION_SET``.

Design principles
-----------------
**Direct label mapping over substring matching**
    The primary mapping table (``_LABEL_MAP``) covers every label produced by
    the target classifier as an exact-match lookup.  This is faster, more
    predictable, and easier to audit than substring search.  Substring
    matching is retained only as a final-resort fallback for unanticipated
    labels.

**Strict schema alignment**
    The only valid output values are the eight members of ``EMOTION_SET`` as
    imported from ``akg/emotion_schema.py``.  No label outside this set can
    appear in any output of this module.

**Determinism**
    ``map_to_occ`` is a pure function with no randomness, no external I/O,
    and no mutable state.  Given the same input string it always returns the
    same output string.

**No-None guarantee**
    ``map_to_occ`` never returns ``None``.  If no mapping matches, the hard
    fallback ``"distress"`` is returned.

References
----------
Hartmann, J. (2022). Emotion English DistilRoBERTa-base.
Ortony, A., Clore, G. L., & Collins, A. (1988). *The Cognitive Structure of
Emotions*. Cambridge University Press.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from akg.emotion_schema import EMOTION_SET

# ---------------------------------------------------------------------------
# Schema anchor
# ---------------------------------------------------------------------------

_VALID_EMOTIONS: frozenset[str] = frozenset(EMOTION_SET)

assert "surprise" not in _VALID_EMOTIONS, (
    "'surprise' must not be a member of EMOTION_SET used for mapping output."
)

# ---------------------------------------------------------------------------
# Direct label mapping
# ---------------------------------------------------------------------------

# Exhaustive mapping from known classifier labels to EMOTION_SET members.
# Covers all output labels of j-hartmann/emotion-english-distilroberta-base
# plus common synonyms and OCC-adjacent terms likely to appear in pipeline
# outputs.  Each value must be a member of EMOTION_SET.

_LABEL_MAP: dict[str, str] = {
    # --- joy ---
    "joy":          "joy",
    "happiness":    "joy",
    "happy":        "joy",
    "pleased":      "joy",
    "delighted":    "joy",
    "excited":      "joy",
    "elation":      "joy",

    # --- distress ---
    "sadness":      "distress",
    "distress":     "distress",
    "sad":          "distress",
    "grief":        "distress",
    "sorrow":       "distress",
    "depressed":    "distress",
    "disappointed": "distress",
    "upset":        "distress",
    "hurt":         "distress",

    # --- fear ---
    "fear":         "fear",
    "anxiety":      "fear",
    "anxious":      "fear",
    "nervous":      "fear",
    "worried":      "fear",
    "scared":       "fear",
    "panic":        "fear",
    "terror":       "fear",

    # --- anger ---
    "anger":        "anger",
    "disgust":      "anger",
    "frustration":  "anger",
    "frustrated":   "anger",
    "annoyed":      "anger",
    "furious":      "anger",
    "rage":         "anger",

    # --- shame ---
    "shame":        "shame",
    "guilt":        "shame",
    "guilty":       "shame",
    "embarrassed":  "shame",
    "ashamed":      "shame",
    "remorse":      "shame",

    # --- gratitude ---
    "gratitude":    "gratitude",
    "love":         "gratitude",
    "thankful":     "gratitude",
    "grateful":     "gratitude",
    "appreciative": "gratitude",
    "admiration":   "gratitude",

    # --- pride ---
    "pride":        "pride",
    "proud":        "pride",
    "confident":    "pride",
    "accomplished": "pride",
    "approval":     "pride",

    # --- hope ---
    "hope":         "hope",
    "hopeful":      "hope",
    "optimistic":   "hope",
    "optimism":     "hope",

    # --- formerly surprise-adjacent labels: remapped to valenced emotions ---
    # These labels describe reactions that resolve to identifiable valenced
    # appraisal states.  Each is mapped to the nearest EMOTION_SET member.
    "amazed":       "joy",       # positive valenced wonder → joy
    "shocked":      "distress",  # negative unexpected event → distress
    "astonished":   "distress",  # negative unexpected event → distress
}

# Compile-time assertion: every value in the map must be in EMOTION_SET.
for _raw, _mapped in _LABEL_MAP.items():
    assert _mapped in _VALID_EMOTIONS, (
        f"_LABEL_MAP entry {_raw!r} -> {_mapped!r} is not in EMOTION_SET. "
        f"Valid: {sorted(_VALID_EMOTIONS)}"
    )

# ---------------------------------------------------------------------------
# Fallback keyword table (last resort for unmapped labels)
# ---------------------------------------------------------------------------

# Applied only when _LABEL_MAP produces no exact match.
# Ordered specific -> general to prevent weaker appraisals overriding
# dominant ones.  Entries are checked as case-insensitive substrings.

_KEYWORD_FALLBACK: list[tuple[str, str]] = [
    ("ashamed",      "shame"),
    ("guilty",       "shame"),
    ("embarrassed",  "shame"),
    ("furious",      "anger"),
    ("annoyed",      "anger"),
    ("angry",        "anger"),
    ("frustrated",   "anger"),
    ("panic",        "fear"),
    ("scared",       "fear"),
    ("anxious",      "fear"),
    ("nervous",      "fear"),
    ("worried",      "fear"),
    ("depressed",    "distress"),
    ("disappointed", "distress"),
    ("grief",        "distress"),
    ("hurt",         "distress"),
    ("sad",          "distress"),
    ("upset",        "distress"),
    ("delighted",    "joy"),
    ("excited",      "joy"),
    ("happy",        "joy"),
    ("pleased",      "joy"),
    ("hopeful",      "hope"),
    ("optimistic",   "hope"),
    ("appreciative", "gratitude"),
    ("grateful",     "gratitude"),
    ("thankful",     "gratitude"),
    ("love",         "gratitude"),
    ("admiration",   "gratitude"),
    ("accomplished", "pride"),
    ("confident",    "pride"),
    ("proud",        "pride"),
]

for _kw, _target in _KEYWORD_FALLBACK:
    assert _target in _VALID_EMOTIONS, (
        f"_KEYWORD_FALLBACK entry {_kw!r} -> {_target!r} is not in EMOTION_SET."
    )

_FALLBACK_EMOTION: str = "distress"
assert _FALLBACK_EMOTION in _VALID_EMOTIONS


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def map_to_occ(label: str, previous_emotion: str | None = None) -> str:
    """Map a classifier output label onto a canonical OCC emotion.

    Uses a three-tier lookup strategy:
    1. Exact match in ``_LABEL_MAP`` (covers all known classifier labels).
    2. Substring keyword scan in ``_KEYWORD_FALLBACK`` (handles unseen labels).
    3. Hard fallback: *previous_emotion* if valid, else ``"distress"``.

    The function never returns ``None`` and never returns a value outside
    the eight members of ``EMOTION_SET``.

    Parameters
    ----------
    label:
        A classifier output label or arbitrary emotion string.
        Case-insensitive; coerced to string if not already.
    previous_emotion:
        The emotion detected in the preceding pipeline step.  Used as the
        contextual fallback when no mapping is found, providing narrative
        continuity.  Must be a member of ``EMOTION_SET`` to be used;
        otherwise the hard fallback ``"distress"`` applies.

    Returns
    -------
    str
        A member of ``EMOTION_SET``:
        ``"joy"``, ``"distress"``, ``"hope"``, ``"fear"``,
        ``"pride"``, ``"shame"``, ``"anger"``, or ``"gratitude"``.

    Examples
    --------
    ::

        >>> map_to_occ("sadness")
        'distress'
        >>> map_to_occ("disgust")
        'anger'
        >>> map_to_occ("love")
        'gratitude'
        >>> map_to_occ("admiration")
        'gratitude'
        >>> map_to_occ("amazed")
        'joy'
        >>> map_to_occ("shocked")
        'distress'
        >>> map_to_occ("unknown_label", previous_emotion="fear")
        'fear'
        >>> map_to_occ("unknown_label")
        'distress'
    """
    normalised: str = str(label).lower().strip()

    # Tier 1: exact match in direct label map.
    if normalised in _LABEL_MAP:
        return _LABEL_MAP[normalised]

    # Tier 2: exact match against EMOTION_SET (handles already-canonical input).
    if normalised in _VALID_EMOTIONS:
        return normalised

    # Tier 3: substring keyword fallback.
    for keyword, occ_emotion in _KEYWORD_FALLBACK:
        if keyword in normalised:
            return occ_emotion

    # Tier 4: contextual fallback — previous emotion if valid, else "distress".
    if previous_emotion is not None and previous_emotion in _VALID_EMOTIONS:
        return previous_emotion

    return _FALLBACK_EMOTION


def is_valid_emotion(label: str) -> bool:
    """Return ``True`` if *label* is a member of ``EMOTION_SET``.

    Performs an exact, case-sensitive membership test against the eight
    canonical OCC emotions.

    Parameters
    ----------
    label:
        The emotion string to test.

    Returns
    -------
    bool

    Examples
    --------
    ::

        >>> is_valid_emotion("anger")
        True
        >>> is_valid_emotion("rage")
        False
        >>> is_valid_emotion("")
        False
    """
    return label in _VALID_EMOTIONS