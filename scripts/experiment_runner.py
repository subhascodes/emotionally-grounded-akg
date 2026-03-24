"""
scripts/emotion_detector.py
============================

Deterministic, traceable emotion detection module for the AKG-constrained
neuro-symbolic storytelling system.

This module maps arbitrary short narrative text onto one of the nine OCC
emotions defined in ``akg/emotion_schema.py``.  It operates as a two-stage
pipeline:

  Stage 1 — NLI inference
      A pre-trained zero-shot classification model (``facebook/bart-large-mnli``)
      scores the semantic entailment between the input text and a set of
      natural-language emotion hypotheses, producing a raw predicted label and
      a confidence score.

  Stage 2 — OCC normalisation
      The raw label is passed through ``akg.emotion_mapping.map_to_occ``,
      which maps it deterministically onto one of the nine canonical OCC
      emotions.  This guarantees that the ``"emotion"`` field of every return
      value is a member of ``EMOTION_LIST``, regardless of what the model
      produces.

Output schema
-------------
Every public function returns dicts conforming to::

    {
        "raw_label":  str,    # label as produced by the classifier
        "emotion":    str,    # OCC-normalised label; always in EMOTION_LIST
        "confidence": float,  # sharpened NLI score for the raw label
    }

Theoretical rationale
---------------------
**Why zero-shot NLI?**
    Supervised classifiers trained on datasets labelled with Ekman or Plutchik
    taxonomies cannot be used directly: their output spaces do not align with
    the OCC appraisal graph.  Zero-shot NLI avoids this by treating each
    candidate label as a natural-language hypothesis, enabling schema-aligned
    inference without retraining.

**Why two-stage (NLI + map_to_occ)?**
    The NLI model may output labels that are valid detection targets but still
    require normalisation (e.g., morphological variants, multi-word phrases).
    Routing all outputs through ``map_to_occ`` provides a deterministic safety
    net that prevents schema violations even if the candidate label set drifts
    from ``EMOTION_LIST``.

**Why is ``"surprise"`` excluded from detection candidates?**
    In OCC theory, surprise is a valence-neutral transient modifier triggered
    by expectation violation.  It is not a dominant sustained appraisal state.
    Excluding it from the NLI hypothesis set forces the model to commit to a
    valenced prediction, which is more actionable for trajectory planning.
    ``"surprise"`` remains reachable in the AKG graph via the symbolic planner.

**Temperature sharpening**
    Raw NLI softmax scores over OCC-adjacent labels tend to be diffuse.
    Sharpening via ``s_i^alpha / sum(s_j^alpha)`` (default ``alpha = 1.5``)
    increases the margin between top-1 and top-2 predictions without altering
    rank order.

**Determinism**
    All inference is performed on CPU with dropout disabled (``model.eval()``
    is set internally by HuggingFace).  Given identical weights and input,
    outputs are reproducible across runs.

Debug mode
----------
Set ``DEBUG_EMOTION=1`` in the environment (or ``.env``) to print a
structured trace block before each prediction::

    [EMOTION DETECTOR]
    Text       : <input text>
    Raw        : <raw_label>
    Mapped     : <emotion>
    Confidence : <confidence>

Zero performance impact when ``DEBUG_EMOTION=0`` (default).

Environment variables
---------------------
::

    DEBUG_EMOTION=0          # set to 1 to enable debug printing
    DEBUG_PRINT_PROMPTS=0    # inherited from story_generator context

Dependencies
------------
* ``transformers`` (HuggingFace, >= 4.30)
* ``torch`` (CPU-only)
* ``akg/emotion_schema.py``
* ``akg/emotion_mapping.py``
* ``python-dotenv``

References
----------
Ortony, A., Clore, G. L., & Collins, A. (1988). *The Cognitive Structure of
Emotions*. Cambridge University Press.
Williams, A., Nangia, N., & Bowman, S. (2018). A broad-coverage challenge
corpus for sentence understanding through inference. *NAACL-HLT*.
Lewis, M., et al. (2020). BART: Denoising sequence-to-sequence pre-training
for natural language generation, translation, and comprehension. *ACL*.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Ensure project root is on sys.path when script is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

_ENV_PATH: Path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH)

# Debug flag — evaluated once at import time; zero cost when disabled.
_DEBUG_EMOTION: bool = os.environ.get("DEBUG_EMOTION", "0").strip() == "1"

# ---------------------------------------------------------------------------
# AKG schema and mapping imports
# ---------------------------------------------------------------------------

from akg.emotion_schema import EMOTION_LIST
from akg.emotion_mapping import map_to_occ, is_valid_emotion

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MODEL_ID: str = "facebook/bart-large-mnli"

# Detection candidate labels: EMOTION_LIST minus "surprise".
# "surprise" is excluded because it is a valence-neutral transient OCC
# modifier, not a dominant sustained appraisal state.  See module docstring.
_DETECTION_LABELS: list[str] = [e for e in EMOTION_LIST if e != "surprise"]

# Default temperature-sharpening exponent.  alpha > 1 suppresses low scores
# relative to high scores, increasing top-1/top-2 margin without altering
# rank order.  alpha = 1.0 recovers the raw NLI softmax distribution.
_DEFAULT_ALPHA: float = 1.5

# Default confidence threshold for the ``is_confident`` flag.
_DEFAULT_THRESHOLD: float = 0.4

# ---------------------------------------------------------------------------
# Singleton model loader
# ---------------------------------------------------------------------------

# Module-level cache: None until first call to _get_pipeline().
_pipeline_instance = None


def _get_pipeline():
    """Return the zero-shot classification pipeline, loading it if necessary.

    Implements a module-level lazy singleton.  The HuggingFace pipeline is
    deserialised exactly once per interpreter session.  Subsequent calls reuse
    the cached instance, avoiding repeated model load overhead.

    The pipeline is pinned to CPU (``device=-1``) for hardware-agnostic
    reproducibility.  HuggingFace sets ``model.eval()`` internally, disabling
    dropout and ensuring deterministic forward passes.

    Returns
    -------
    transformers.ZeroShotClassificationPipeline
        A ready-to-use NLI-based classifier backed by
        ``facebook/bart-large-mnli``.
    """
    global _pipeline_instance
    if _pipeline_instance is None:
        # Deferred import: allows the module to be imported without torch
        # in environments where inference is mocked or not required.
        from transformers import pipeline as hf_pipeline

        _pipeline_instance = hf_pipeline(
            task="zero-shot-classification",
            model=_MODEL_ID,
            device=-1,  # CPU; change to device=0 for GPU
        )
    return _pipeline_instance


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _classify(text: str) -> dict:
    """Run NLI inference for a single validated text string.

    Single point of contact with the HuggingFace pipeline.  Returns raw
    output only; no sharpening, threshold logic, or OCC mapping applied here.

    Parameters
    ----------
    text:
        A non-empty, pre-stripped string to classify.

    Returns
    -------
    dict
        Raw HuggingFace output:
        ``{"sequence": str, "labels": list[str], "scores": list[float]}``.
        Labels are sorted by score in descending order.
    """
    classifier = _get_pipeline()
    return classifier(
        sequences=text,
        candidate_labels=_DETECTION_LABELS,
        multi_label=False,  # mutually exclusive softmax over all candidates
    )


def _sharpen(scores: list[float], alpha: float) -> list[float]:
    """Apply temperature sharpening and renormalise a score distribution.

    Exponentiates each score by *alpha* and divides by the sum, producing a
    probability distribution with enhanced contrast between high and low
    scoring labels.

    Parameters
    ----------
    scores:
        Raw softmax scores.  Must all be non-negative.
    alpha:
        Sharpening exponent.  Must be > 0.

    Returns
    -------
    list[float]
        Sharpened, renormalised scores in the same order as *scores*.
    """
    sharpened = [s ** alpha for s in scores]
    total = sum(sharpened)
    if total == 0.0:
        raise ValueError(
            "All sharpened scores are zero; cannot renormalise."
        )
    return [s / total for s in sharpened]


def _build_result(raw: dict, threshold: float, alpha: float) -> dict:
    """Convert raw NLI output into the AKG two-stage result schema.

    Applies temperature sharpening to raw scores, extracts the top prediction,
    maps it through ``map_to_occ`` to guarantee OCC schema compliance, and
    assembles the output dictionary.

    Parameters
    ----------
    raw:
        Output dict from :func:`_classify`.  Labels pre-sorted by raw score
        descending.
    threshold:
        Confidence threshold for the ``is_confident`` flag.
    alpha:
        Temperature-sharpening exponent.

    Returns
    -------
    dict
        ``{"raw_label": str, "emotion": str, "confidence": float,
           "all_scores": dict[str, float], "is_confident": bool}``

        * ``raw_label``  — top NLI label before OCC normalisation.
        * ``emotion``    — OCC-normalised label; always in ``EMOTION_LIST``.
        * ``confidence`` — sharpened top-1 score, rounded to six decimals.
        * ``all_scores`` — full sharpened distribution keyed by label.
        * ``is_confident`` — ``True`` iff ``confidence >= threshold``.
    """
    labels: list[str] = raw["labels"]
    raw_scores: list[float] = raw["scores"]

    sharpened: list[float] = _sharpen(raw_scores, alpha)

    # Labels remain sorted by raw score; sharpening is rank-preserving.
    raw_label: str = labels[0]
    top_score: float = sharpened[0]

    # Stage 2: OCC normalisation via map_to_occ.
    # Guarantees "emotion" is always a member of EMOTION_LIST.
    occ_emotion: str = map_to_occ(raw_label)

    all_scores: dict[str, float] = {
        label: round(score, 6)
        for label, score in zip(labels, sharpened)
    }

    return {
        "raw_label":    raw_label,
        "emotion":      occ_emotion,
        "confidence":   round(top_score, 6),
        "all_scores":   all_scores,
        "is_confident": top_score >= threshold,
    }


def _validate_text(text: str) -> str:
    """Validate and normalise a text input.

    Parameters
    ----------
    text:
        Raw input string from the caller.

    Returns
    -------
    str
        Stripped, non-empty string ready for inference.

    Raises
    ------
    TypeError
        If *text* is not a string.
    ValueError
        If *text* is empty or whitespace-only after stripping.
    """
    if not isinstance(text, str):
        raise TypeError(
            f"text must be a string, got {type(text).__name__!r}."
        )
    stripped = text.strip()
    if not stripped:
        raise ValueError("text must not be empty or whitespace-only.")
    return stripped


def _validate_threshold(threshold: float) -> None:
    """Assert that *threshold* is a numeric value in the open interval (0, 1).

    Raises
    ------
    TypeError / ValueError
        On invalid type or out-of-range value.
    """
    if not isinstance(threshold, (float, int)) or isinstance(threshold, bool):
        raise TypeError(
            f"threshold must be a float, got {type(threshold).__name__!r}."
        )
    if not (0.0 < threshold < 1.0):
        raise ValueError(
            f"threshold must be in (0, 1), got {threshold}."
        )


def _validate_alpha(alpha: float) -> None:
    """Assert that *alpha* is a positive numeric value.

    Raises
    ------
    TypeError / ValueError
        On invalid type or non-positive value.
    """
    if not isinstance(alpha, (float, int)) or isinstance(alpha, bool):
        raise TypeError(
            f"alpha must be a float, got {type(alpha).__name__!r}."
        )
    if alpha <= 0.0:
        raise ValueError(f"alpha must be > 0, got {alpha}.")


def _maybe_debug(text: str, result: dict) -> None:
    """Print a structured debug trace when ``DEBUG_EMOTION=1``.

    Zero cost when debug mode is disabled: the flag is evaluated at import
    time and the function body returns immediately without any string
    formatting.

    Parameters
    ----------
    text:
        The original input text passed to the detector.
    result:
        The completed result dictionary returned by the detector.
    """
    if not _DEBUG_EMOTION:
        return
    print(
        f"[EMOTION DETECTOR]\n"
        f"Text       : {text}\n"
        f"Raw        : {result['raw_label']}\n"
        f"Mapped     : {result['emotion']}\n"
        f"Confidence : {result['confidence']:.4f}"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_emotion(
    text: str,
    threshold: float = _DEFAULT_THRESHOLD,
    alpha: float = _DEFAULT_ALPHA,
) -> dict:
    """Detect the dominant OCC emotion in a short narrative text.

    Two-stage pipeline:
    1. Zero-shot NLI inference produces a raw emotion label and score.
    2. ``map_to_occ`` normalises the raw label to a canonical OCC emotion.

    The returned ``"emotion"`` field is always a member of ``EMOTION_LIST``,
    regardless of what the NLI model outputs.

    Parameters
    ----------
    text:
        A short narrative sentence or passage (typically 1-3 sentences).
    threshold:
        Minimum sharpened confidence score required to set
        ``is_confident = True``.  Does not alter predicted labels.
        Must be in ``(0, 1)``.  Defaults to ``0.4``.
    alpha:
        Temperature-sharpening exponent applied to raw NLI scores.
        ``alpha = 1.0`` returns unmodified scores.  ``alpha > 1.0``
        increases top-1/top-2 discrimination.  Must be > 0.
        Defaults to ``1.5``.

    Returns
    -------
    dict with the following keys:

    ``raw_label`` : str
        The emotion label as returned by the NLI classifier before any
        OCC normalisation.  May differ from ``"emotion"`` if ``map_to_occ``
        performs a remapping (rare when candidate labels are drawn from
        ``EMOTION_LIST``).

    ``emotion`` : str
        OCC-normalised emotion label.  Always a member of ``EMOTION_LIST``.

    ``confidence`` : float
        Sharpened top-1 entailment score, rounded to six decimal places.

    ``all_scores`` : dict[str, float]
        Full sharpened score distribution over all detection candidates.
        ``"surprise"`` is absent (excluded from detection).

    ``is_confident`` : bool
        ``True`` iff ``confidence >= threshold``.

    Raises
    ------
    TypeError
        If *text* is not a string, *threshold* or *alpha* are not numeric.
    ValueError
        If *text* is empty, *threshold* is outside ``(0, 1)``, or *alpha*
        is <= 0.

    Examples
    --------
    ::

        >>> result = detect_emotion("She felt a wave of shame wash over her.")
        >>> result["raw_label"]
        'shame'
        >>> result["emotion"]
        'shame'
        >>> result["is_confident"]
        True

        >>> result = detect_emotion("He was furious at the betrayal.")
        >>> result["emotion"]
        'anger'
    """
    _validate_threshold(threshold)
    _validate_alpha(alpha)
    clean_text = _validate_text(text)

    raw = _classify(clean_text)
    result = _build_result(raw, threshold, alpha)

    _maybe_debug(text, result)
    return result


def detect_batch(
    texts: list[str],
    threshold: float = _DEFAULT_THRESHOLD,
    alpha: float = _DEFAULT_ALPHA,
) -> list[dict]:
    """Detect OCC emotions in a list of text strings.

    Applies :func:`detect_emotion` independently to each element of *texts*.
    Each result is independent: scores are not affected by batch composition.
    This guarantees output equivalence with element-wise :func:`detect_emotion`
    calls, which is required for evaluation reproducibility.

    Parameters
    ----------
    texts:
        Ordered list of narrative text strings.  An empty list returns an
        empty list without loading the model.
    threshold:
        Confidence threshold applied uniformly.  Must be in ``(0, 1)``.
        Defaults to ``0.4``.
    alpha:
        Temperature-sharpening exponent applied uniformly.  Must be > 0.
        Defaults to ``1.5``.

    Returns
    -------
    list[dict]
        Result dictionaries in input order.  Each dict matches the schema
        returned by :func:`detect_emotion`.

    Raises
    ------
    TypeError
        If *texts* is not a list, any element is not a string, or numeric
        parameters are of the wrong type.
    ValueError
        If any element is empty, *threshold* is outside ``(0, 1)``, or
        *alpha* is <= 0.

    Examples
    --------
    ::

        >>> results = detect_batch([
        ...     "He finally won the championship.",
        ...     "She was terrified of what lay ahead.",
        ... ])
        >>> [r["emotion"] for r in results]
        ['joy', 'fear']
        >>> all(r["raw_label"] for r in results)
        True
    """
    if not isinstance(texts, list):
        raise TypeError(
            f"texts must be a list, got {type(texts).__name__!r}."
        )
    _validate_threshold(threshold)
    _validate_alpha(alpha)

    return [detect_emotion(text=t, threshold=threshold, alpha=alpha) for t in texts]


# ---------------------------------------------------------------------------
# CLI demo entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _DEMO_CASES: list[tuple[str, str]] = [
        (
            "She finally received the letter confirming her scholarship — "
            "everything she had worked for had paid off.",
            "joy",
        ),
        (
            "He discovered that his trusted colleague had been sabotaging "
            "his work for months, and he could barely contain his rage.",
            "anger",
        ),
        (
            "Standing at the edge of the dark alley, she heard footsteps "
            "behind her and felt her heart begin to race.",
            "fear",
        ),
        (
            "He had humiliated her in front of everyone, yet she had stayed "
            "silent — now the memory burned with self-reproach.",
            "shame",
        ),
        (
            "The test results were not yet back, but there was still a chance "
            "the treatment had worked.",
            "hope",
        ),
    ]

    _THRESHOLD: float = _DEFAULT_THRESHOLD
    _ALPHA: float = _DEFAULT_ALPHA

    print("=" * 70)
    print("AKG Emotion Detector — CLI Demo")
    print(f"Model           : {_MODEL_ID}")
    print(f"Detection labels: {_DETECTION_LABELS}")
    print(f"Excluded        : ['surprise']  (transient OCC modifier)")
    print(f"Threshold       : {_THRESHOLD}  |  Alpha: {_ALPHA}")
    print("=" * 70)

    _texts = [sentence for sentence, _ in _DEMO_CASES]
    _results = detect_batch(_texts, threshold=_THRESHOLD, alpha=_ALPHA)

    for (sentence, expected), result in zip(_DEMO_CASES, _results):
        correct_marker = "V" if result["emotion"] == expected else "X"
        raw_match = "" if result["raw_label"] == result["emotion"] else f"  (raw: {result['raw_label']})"
        print(f"\nInput      : {sentence}")
        print(f"Expected   : {expected}")
        print(
            f"Predicted  : {result['emotion']}{raw_match}  "
            f"(confidence: {result['confidence']:.4f}, "
            f"is_confident: {result['is_confident']})  [{correct_marker}]"
        )
        sorted_scores = sorted(
            result["all_scores"].items(), key=lambda x: x[1], reverse=True
        )
        print("All scores (sharpened):")
        for label, score in sorted_scores:
            bar = "|" * int(score * 30)
            marker = " <- predicted" if label == result["raw_label"] else ""
            print(f"  {label:12s} {score:.4f}  {bar}{marker}")

    _correct = sum(
        1 for (_, exp), res in zip(_DEMO_CASES, _results)
        if res["emotion"] == exp
    )
    _avg_confidence = sum(r["confidence"] for r in _results) / len(_results)

    print("\n" + "=" * 70)
    print(f"Demo accuracy    : {_correct}/{len(_DEMO_CASES)}")
    print(f"Avg confidence   : {_avg_confidence:.4f}")
    print(f"Schema compliant : {all(is_valid_emotion(r['emotion']) for r in _results)}")
    print("=" * 70)