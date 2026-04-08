"""
scripts/emotion_detector.py

Deterministic emotion detection module for neuro-symbolic storytelling system.
Maps raw text to OCC emotions using a hybrid pipeline:

    1. Lexical pre-check  — fast keyword scan; if a strong signal is found, the
                            emotion is returned immediately with boosted confidence
                            without invoking the transformer.
    2. Transformer model  — j-hartmann/emotion-english-distilroberta-base via
                            HuggingFace pipeline.
    3. Top-k OCC mapping  — top-3 raw labels mapped through map_to_occ.
    4. Rule-based layer   — keyword heuristics applied over mapped candidates.
    5. Fallback chain     — previous_emotion → "distress".

Decision priority:
    lexical match  >  rule match  >  model top-1  >  fallback
"""

import os
import torch
from transformers import pipeline

from akg.emotion_schema import EMOTION_SET, is_valid_emotion
from akg.emotion_mapping import map_to_occ

# ── Determinism ──────────────────────────────────────────────────────────────
torch.manual_seed(0)

# ── Constants ────────────────────────────────────────────────────────────────
MODEL_ID   = "j-hartmann/emotion-english-distilroberta-base"
DEBUG_ENV  = "DEBUG_EMOTION"

# ── Lexical confidence boost ──────────────────────────────────────────────────
_LEXICAL_CONFIDENCE: float = 0.92   # ≥ 0.9 as required

# ── Lexical signal dictionary ─────────────────────────────────────────────────
# Maps each OCC emotion to a list of strong surface-level keyword signals.
# A hit on any keyword triggers an immediate return, bypassing the transformer.
LEXICAL_SIGNALS: dict = {
    "shame":     ["ashamed", "embarrassed", "humiliated", "guilty", "regret"],
    "anger":     ["angry", "furious", "rage", "slammed", "yelled"],
    "fear":      ["afraid", "terrified", "scared", "panic"],
    "hope":      ["hope", "chance", "maybe", "believe"],
    "joy":       ["happy", "excited", "delighted", "thrilled"],
    "distress":  ["sad", "hurt", "upset", "pain"],
    "pride":     ["proud", "accomplished"],
    "gratitude": ["grateful", "thankful"],
}

# ── Singleton pipeline ────────────────────────────────────────────────────────
_pipeline_instance = None


def _get_pipeline():
    """Load and return the singleton HuggingFace classification pipeline."""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = pipeline(
            "text-classification",
            model=MODEL_ID,
            top_k=None,
            device=-1,  # CPU only
        )
    return _pipeline_instance


# ── Helper: input validation ──────────────────────────────────────────────────

def _validate_text(text):
    """
    Validate that input text is a non-empty string.

    Raises:
        TypeError:  if text is not a str.
        ValueError: if text is empty or whitespace-only.
    """
    if not isinstance(text, str):
        raise TypeError(
            f"Input text must be a string, got {type(text).__name__!r}."
        )
    if not text.strip():
        raise ValueError("Input text must not be empty or whitespace-only.")


# ── Helper: lexical pre-check ─────────────────────────────────────────────────

def _lexical_check(text: str) -> tuple:
    """
    Scan the text for strong keyword signals before invoking the transformer.

    Iterates through LEXICAL_SIGNALS in dictionary-insertion order and returns
    on the first matching keyword, making the check deterministic.

    Args:
        text (str): Input text (lowercased internally).

    Returns:
        tuple[str, float] | tuple[None, None]:
            (matched_emotion, _LEXICAL_CONFIDENCE) on a hit, or
            (None, None) when no keyword is found.
    """
    t = text.lower()
    for emotion, keywords in LEXICAL_SIGNALS.items():
        for kw in keywords:
            if kw in t:
                return emotion, _LEXICAL_CONFIDENCE
    return None, None


# ── Helper: transformer classify ─────────────────────────────────────────────

def _classify(text: str) -> list:
    """
    Run the HuggingFace pipeline on a single text string.

    Returns:
        list[dict]: Top-3 results sorted descending by score, each with
                    keys 'label' (str, lowercased) and 'score' (float).
    """
    classifier = _get_pipeline()
    results = classifier(text)

    # pipeline with top_k=None returns [[{label, score}, ...]]
    if isinstance(results[0], list):
        scores = results[0]
    else:
        scores = results

    scores_sorted = sorted(scores, key=lambda x: x["score"], reverse=True)

    return [
        {"label": r["label"].lower(), "score": float(r["score"])}
        for r in scores_sorted[:3]
    ]


# ── Helper: rule-based correction layer ──────────────────────────────────────

def _apply_rules(text: str, candidates: list) -> str:
    """
    Apply keyword-based heuristics to correct or confirm the mapped emotion.

    This layer operates on the OCC-mapped top-k candidates after the
    transformer has already run.  Rules are evaluated in priority order;
    the first match wins.  Returns '' when no rule fires.

    Args:
        text       (str):       Original input text (lowercased internally).
        candidates (list[str]): OCC-mapped emotions for the top-k raw labels.

    Returns:
        str: Matched OCC emotion, or '' if no rule matched.
    """
    t = text.lower()

    if any(w in t for w in [
        "ashamed", "embarrassed", "humiliated",
        "face burning", "couldn't look", "avoided eye contact",
    ]):
        return "shame"

    if any(w in t for w in ["angry", "rage", "furious", "yelled", "slammed"]):
        return "anger"

    if any(w in t for w in ["hope", "maybe", "chance", "could still"]):
        return "hope"

    if any(w in t for w in ["afraid", "terrified", "panic", "danger"]):
        return "fear"

    return ""


# ── Helper: debug printer ─────────────────────────────────────────────────────

def _maybe_debug(
    text: str,
    source: str,
    raw_labels,
    mapped_labels,
    final_emotion: str,
    confidence: float,
) -> None:
    """Print debug information when DEBUG_EMOTION=1 is set."""
    if os.environ.get(DEBUG_ENV, "0") == "1":
        print(
            f"[EMOTION DETECTOR]  "
            f"Text: {text!r}  "
            f"Source: {source}  "
            f"Raw labels: {raw_labels}  "
            f"Mapped labels: {mapped_labels}  "
            f"Final emotion: {final_emotion}  "
            f"Confidence: {confidence:.4f}"
        )


# ── Core detection ────────────────────────────────────────────────────────────

def detect_emotion(text, previous_emotion=None):
    """
    Detect the OCC emotion for a single text input using a hybrid pipeline.

    Pipeline:
        1. Validate input text.
        2. Lexical pre-check — if a strong keyword signal is found, return
           immediately with _LEXICAL_CONFIDENCE (bypasses transformer).
        3. Run HuggingFace classifier; retrieve top-3 raw labels + scores.
        4. Map all top-3 labels to OCC emotions via map_to_occ.
        5. Apply rule-based correction layer (_apply_rules).
        6. Hybrid decision: lexical > rule > model top-1.
        7. Validate; apply dual-stage fallback if invalid.

    Args:
        text             (str):       Input text to analyse.
        previous_emotion (str|None):  Previously active OCC emotion (for
                                      context-aware mapping and fallback).

    Returns:
        dict: {"emotion": str, "confidence": float}
    """
    # Step 1 – validate
    _validate_text(text)

    # Step 2 – lexical pre-check (fast path)
    lex_emotion, lex_confidence = _lexical_check(text)

    if lex_emotion is not None and is_valid_emotion(lex_emotion):
        _maybe_debug(text, "lexical", [], [], lex_emotion, lex_confidence)
        return {
            "emotion":    lex_emotion,
            "confidence": round(lex_confidence, 6),
        }

    # Step 3 – transformer classification (slow path)
    top_k      = _classify(text)
    confidence = top_k[0]["score"]
    raw_labels = [r["label"] for r in top_k]

    # Step 4 – map top-k raw labels to OCC emotions
    mapped = [map_to_occ(r["label"].lower(), previous_emotion) for r in top_k]

    # Step 5 – rule-based correction
    rule_emotion = _apply_rules(text, mapped)

    # Step 6 – hybrid decision: rule > model top-1
    if rule_emotion and is_valid_emotion(rule_emotion):
        final = rule_emotion
        source = "rule"
    else:
        final = mapped[0]
        source = "model"

    # Step 7 – validate + dual-stage fallback
    if not is_valid_emotion(final):
        if previous_emotion is not None and is_valid_emotion(previous_emotion):
            final  = previous_emotion
            source = "fallback:previous"
        else:
            final  = "distress"
            source = "fallback:default"

    _maybe_debug(text, source, raw_labels, mapped, final, confidence)

    return {
        "emotion":    final,
        "confidence": round(confidence, 6),
    }


# ── Batch detection ───────────────────────────────────────────────────────────

def detect_batch(texts, previous_emotions=None):
    """
    Detect OCC emotions for a list of text inputs.

    Args:
        texts             (list[str]):       List of input texts.
        previous_emotions (list[str]|None):  Parallel list of previous OCC
                                             emotions (or None to skip context).

    Returns:
        list[dict]: Each element is {"emotion": str, "confidence": float}.
    """
    if not isinstance(texts, (list, tuple)):
        raise TypeError(
            f"'texts' must be a list or tuple, got {type(texts).__name__!r}."
        )

    if previous_emotions is None:
        previous_emotions = [None] * len(texts)

    if len(previous_emotions) != len(texts):
        raise ValueError(
            "'previous_emotions' must have the same length as 'texts'."
        )

    return [
        detect_emotion(text, previous_emotion=prev)
        for text, prev in zip(texts, previous_emotions)
    ]


# ── CLI test block ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    TEST_CASES = [
        # (text, expected_emotion)
        ("I just got promoted! This is the best day of my life!", "joy"),
        ("I am terrified about what might happen next.", "fear"),
        ("He felt ashamed — he couldn't look anyone in the eye after what he'd done.", "shame"),
        ("Despite everything, she believed there was still a chance things would turn around.", "hope"),
        ("She was furious, slamming her fist on the desk as the words poured out.", "anger"),
        ("I am so grateful for everything you have done for me.", "gratitude"),
        ("She felt proud of what she had accomplished against all the odds.", "pride"),
        ("He was hurt and upset, the pain of their words cutting deep.", "distress"),
    ]

    print("=" * 60)
    print("EMOTION DETECTOR — CLI TEST (lexical + model hybrid)")
    print("=" * 60)

    correct = 0
    for idx, (text, expected) in enumerate(TEST_CASES, start=1):
        result     = detect_emotion(text, previous_emotion=None)
        emotion    = result["emotion"]
        confidence = result["confidence"]
        match      = "✓" if emotion == expected else "✗"
        if emotion == expected:
            correct += 1
        print(
            f"[{idx}] {match}  "
            f"Text      : {text}\n"
            f"    Expected  : {expected}\n"
            f"    Predicted : {emotion}  (confidence={confidence:.4f})\n"
        )

    accuracy = correct / len(TEST_CASES) * 100
    print("-" * 60)
    print(f"Accuracy: {correct}/{len(TEST_CASES)} = {accuracy:.1f}%")
    print("=" * 60)