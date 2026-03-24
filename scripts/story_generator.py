"""
scripts/story_generator.py
===========================

Controlled narrative generation layer for the AKG-constrained storytelling
system.

This module is the core integration point of the prototype.  It coordinates
the symbolic emotional planner, AKG transition metadata, LLM backend, emotion
detector, and transition validator to produce emotionally grounded story
continuations under four experimental conditions.

Experimental modes
------------------
The four modes form a strict ablation ladder, each adding one component of
the full neuro-symbolic pipeline:

``baseline_free``
    Uncontrolled generation with no emotional guidance.  Establishes the
    performance floor: ETVS measures how often an unconstrained LLM happens
    to produce AKG-valid transitions.

``baseline_sequence_prompt``
    Soft emotional guidance via a single sequence-level prompt.  Tests whether
    prompting with an emotion trajectory — without per-step enforcement —
    improves ETVS over the free baseline.

``baseline_planner_no_validation``
    Per-step symbolic guidance with AKG metadata injection but no mismatch
    correction.  Isolates the effect of appraisal-grounded prompting from the
    effect of the validation-retry loop.

``full_model``
    Full neuro-symbolic control: per-step guidance, detected emotion
    verification, and retry on mismatch.  Establishes the performance ceiling.

Return schema
-------------
All modes return a dict with the same keys, enabling direct cross-mode
comparison::

    {
        "mode":               str,
        "seed_emotion":       str,
        "planned_trajectory": list[str],
        "realized_emotions":  list[str],
        "generated_segments": list[str],
        "etvs":               float,
        "retry_counts":       list[int],
        "llm_call_count":     int,
    }

``retry_counts`` is a list of length ``k-1`` (number of transitions).  For
modes that do not retry, every entry is 0.

``llm_call_count`` is the total number of ``generate_text`` calls made by
this mode invocation, including retries.

Stability features
------------------
* A ``0.3 s`` sleep is inserted after every ``generate_text`` call to reduce
  burst pressure on the API beyond the per-call throttle in ``llm_backend``.
* The retry loop in ``full_model`` appends a reinforcement line to the prompt
  on each retry attempt, making the target emotion more explicit without
  altering the planned symbolic trajectory.
* Retry count is capped at ``MAX_RETRIES`` (from ``.env``); the loop never
  spins beyond this bound.
* Sentence segmentation in baseline modes uses a regex boundary split rather
  than naive character replacement, preserving punctuation context.

Environment variables (``.env``)
----------------------------------
::

    MAX_RETRIES=3
    PLANNER_SEED=42

Dependencies
------------
* ``scripts/llm_backend.py``
* ``scripts/emotion_planner.py``
* ``scripts/emotion_detector.py``
* ``scripts/explanation_engine.py``
* ``akg/transition_validator.py``
* ``python-dotenv``
"""

from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# Ensure project root is on sys.path when script is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

_ENV_PATH: Path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH)

MAX_RETRIES: int = int(os.environ.get("MAX_RETRIES", "3"))
PLANNER_SEED: int = int(os.environ.get("PLANNER_SEED", "42"))

# Post-generate-text sleep to further reduce burst API load.
# This is in addition to the per-call delay inside llm_backend._generate_groq.
_POST_GENERATE_SLEEP: float = 0.3

# ---------------------------------------------------------------------------
# Internal imports (after sys.path is set)
# ---------------------------------------------------------------------------

from scripts.llm_backend import generate_text
from scripts.emotion_planner import plan_emotion_trajectory
from scripts.emotion_detector import detect_emotion
from scripts.explanation_engine import get_transition_metadata
from akg.transition_validator import validate_sequence

# ---------------------------------------------------------------------------
# Valid modes
# ---------------------------------------------------------------------------

_VALID_MODES: frozenset[str] = frozenset({
    "baseline_free",
    "baseline_sequence_prompt",
    "baseline_planner_no_validation",
    "full_model",
})

# ---------------------------------------------------------------------------
# Prompt builders (pure functions, no side effects)
# ---------------------------------------------------------------------------

def _prompt_free(seed_text: str, k: int) -> str:
    """Build an unconstrained story continuation prompt.

    Parameters
    ----------
    seed_text:
        The opening narrative passage.
    k:
        Target story length in emotional steps; used to calibrate the
        requested number of continuation sentences.

    Returns
    -------
    str
        Prompt string ready for ``generate_text``.
    """
    target_sentences = max(2, k * 2)
    return (
        f"Continue the following story naturally in approximately "
        f"{target_sentences} sentences. "
        f"Write only the story continuation, no commentary.\n\n"
        f"Story so far:\n{seed_text}"
    )


def _prompt_sequence(seed_text: str, trajectory: list[str]) -> str:
    """Build a sequence-level emotionally guided prompt.

    Parameters
    ----------
    seed_text:
        The opening narrative passage.
    trajectory:
        Planned emotion sequence including the seed emotion as first element.

    Returns
    -------
    str
        Prompt string ready for ``generate_text``.
    """
    arrow_sequence = " -> ".join(trajectory)
    target_sentences = max(2, (len(trajectory) - 1) * 2)
    return (
        f"Continue the following story in approximately {target_sentences} "
        f"sentences so that the character's emotional state progresses through "
        f"this exact sequence: {arrow_sequence}.\n"
        f"Write only the story continuation, no commentary.\n\n"
        f"Story so far:\n{seed_text}"
    )


def _prompt_step(
    seed_text: str,
    src_emotion: str,
    tgt_emotion: str,
    appraisal_condition: str,
    behavioral_tendency: str,
) -> str:
    """Build a single-step AKG-metadata-grounded prompt.

    Parameters
    ----------
    seed_text:
        The story text accumulated so far (seed + all accepted segments).
    src_emotion:
        The emotion the character is currently experiencing.
    tgt_emotion:
        The target emotion for this narrative step.
    appraisal_condition:
        OCC appraisal condition from the AKG edge metadata.
    behavioral_tendency:
        Action-tendency description from the AKG edge metadata.

    Returns
    -------
    str
        Prompt string ready for ``generate_text``.
    """
    return (
        f"Continue the following story in 1-2 sentences.\n"
        f"The character currently feels: {src_emotion}.\n"
        f"They should transition to feeling: {tgt_emotion}.\n"
        f"This shift happens because: {appraisal_condition}.\n"
        f"The character's response should involve: {behavioral_tendency}.\n"
        f"Write only the story continuation, no commentary.\n\n"
        f"Story so far:\n{seed_text}"
    )


def _prompt_step_retry(base_prompt: str, tgt_emotion: str) -> str:
    """Append a reinforcement line to *base_prompt* for a retry attempt.

    The symbolic trajectory is unchanged; only the surface instruction is
    made more explicit to improve the probability of the LLM expressing the
    target emotion unambiguously.

    Parameters
    ----------
    base_prompt:
        The original step prompt produced by ``_prompt_step``.
    tgt_emotion:
        The target emotion that was not realized in the previous attempt.

    Returns
    -------
    str
        Augmented prompt with reinforcement suffix appended.
    """
    reinforcement = (
        f"\nThe previous attempt did not clearly express {tgt_emotion}. "
        f"Make the emotional shift to {tgt_emotion} more explicit."
    )
    return base_prompt + reinforcement


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _detect(text: str) -> str:
    """Detect the dominant OCC emotion in *text*.

    Wraps ``detect_emotion`` and extracts only the predicted label string,
    providing a minimal interface for use within generation loops.

    Parameters
    ----------
    text:
        A short narrative passage to classify.

    Returns
    -------
    str
        Predicted OCC emotion label.
    """
    return detect_emotion(text)["emotion"]


def _plan(seed_emotion: str, k: int) -> list[str]:
    """Generate a k-step AKG-valid trajectory from *seed_emotion*.

    Uses the module-level ``PLANNER_SEED`` for reproducibility.

    Parameters
    ----------
    seed_emotion:
        Starting emotion node.
    k:
        Total trajectory length including the seed.

    Returns
    -------
    list[str]
        Ordered list of *k* emotion strings.
    """
    return plan_emotion_trajectory(
        start_emotion=seed_emotion,
        k=k,
        seed=PLANNER_SEED,
    )


def _fetch_meta(src: str, tgt: str) -> dict[str, str]:
    """Retrieve AKG edge metadata, falling back to empty strings on miss.

    Parameters
    ----------
    src:
        Source emotion node name.
    tgt:
        Target emotion node name.

    Returns
    -------
    dict[str, str]
        Dictionary with ``"appraisal_condition"`` and
        ``"behavioral_tendency"``.  Values are empty strings if the edge is
        not found in Neo4j (should not occur for planner-generated pairs, but
        handled defensively).
    """
    meta = get_transition_metadata(src, tgt)
    if meta is None:
        return {"appraisal_condition": "", "behavioral_tendency": ""}
    return meta


def _compute_etvs(realized: list[str]) -> float:
    """Compute ETVS over the realized emotion sequence.

    Parameters
    ----------
    realized:
        List of detected emotion strings.

    Returns
    -------
    float
        ETVS in ``[0.0, 1.0]``.  Returns 1.0 for single-element sequences.
    """
    if len(realized) < 1:
        return 1.0
    result = validate_sequence(realized)
    return result["etvs"]


def _segment_text(text: str, n_segments: int) -> list[str]:
    """Split *text* into *n_segments* roughly equal chunks using regex boundaries.

    Uses ``re.split(r'(?<=[.!?])\\s+', text.strip())`` to split on sentence
    boundaries, preserving punctuation.  Chunks are re-joined with a space.
    Falls back to the full text as a single segment if splitting produces
    too few sentences.

    Parameters
    ----------
    text:
        Full continuation string to segment.
    n_segments:
        Number of chunks to produce.  Must be >= 1.

    Returns
    -------
    list[str]
        List of *n_segments* non-empty strings.  If segmentation yields
        fewer items than requested, the last available chunk is repeated to
        pad the list.
    """
    if n_segments < 1:
        return [text]

    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()]

    if not sentences:
        return [text] * n_segments

    chunk_size = max(1, len(sentences) // n_segments)
    segments: list[str] = []
    for i in range(n_segments):
        start = i * chunk_size
        end = start + chunk_size if i < n_segments - 1 else len(sentences)
        chunk = " ".join(sentences[start:end]).strip()
        segments.append(chunk if chunk else text)

    # Pad if fewer segments than requested.
    while len(segments) < n_segments:
        segments.append(segments[-1])

    return segments


def _guarded_generate(prompt: str, llm_call_counter: list[int]) -> str:
    """Call ``generate_text`` and apply post-call sleep.

    Increments *llm_call_counter[0]* by 1 on each call so callers can track
    total LLM invocations without mutable shared state.

    Parameters
    ----------
    prompt:
        Full prompt string.
    llm_call_counter:
        Single-element list used as a mutable integer counter.

    Returns
    -------
    str
        Generated text from the LLM backend.
    """
    text = generate_text(prompt)
    llm_call_counter[0] += 1
    time.sleep(_POST_GENERATE_SLEEP)
    return text


# ---------------------------------------------------------------------------
# Mode implementations
# ---------------------------------------------------------------------------

def _run_baseline_free(seed_text: str, k: int) -> dict:
    """Execute the ``baseline_free`` mode.

    Generates a single unconstrained continuation, then detects emotions in
    each generated sentence segment to build the realized sequence.

    Parameters
    ----------
    seed_text:
        Opening narrative passage.
    k:
        Number of emotional steps (sentences generated ~ k*2).

    Returns
    -------
    dict
        Standardised result dictionary.
    """
    seed_emotion = _detect(seed_text)
    trajectory = _plan(seed_emotion, k)  # computed for length alignment only
    n_transitions = k - 1
    llm_calls = [0]

    prompt = _prompt_free(seed_text, k)
    full_continuation = _guarded_generate(prompt, llm_calls)

    segments = _segment_text(full_continuation, max(1, n_transitions))

    realized: list[str] = [seed_emotion]
    for seg in segments:
        realized.append(_detect(seg))

    return {
        "mode": "baseline_free",
        "seed_emotion": seed_emotion,
        "planned_trajectory": trajectory,
        "realized_emotions": realized,
        "generated_segments": segments,
        "etvs": _compute_etvs(realized),
        "retry_counts": [0] * n_transitions,
        "llm_call_count": llm_calls[0],
    }


def _run_baseline_sequence_prompt(seed_text: str, k: int) -> dict:
    """Execute the ``baseline_sequence_prompt`` mode.

    Plans a trajectory and encodes it into a single sequence-level prompt.
    Generates once; detects realized emotions per segment.

    Parameters
    ----------
    seed_text:
        Opening narrative passage.
    k:
        Total trajectory length.

    Returns
    -------
    dict
        Standardised result dictionary.
    """
    seed_emotion = _detect(seed_text)
    trajectory = _plan(seed_emotion, k)
    n_transitions = k - 1
    llm_calls = [0]

    prompt = _prompt_sequence(seed_text, trajectory)
    full_continuation = _guarded_generate(prompt, llm_calls)

    segments = _segment_text(full_continuation, max(1, n_transitions))

    realized: list[str] = [seed_emotion]
    for seg in segments:
        realized.append(_detect(seg))

    return {
        "mode": "baseline_sequence_prompt",
        "seed_emotion": seed_emotion,
        "planned_trajectory": trajectory,
        "realized_emotions": realized,
        "generated_segments": segments,
        "etvs": _compute_etvs(realized),
        "retry_counts": [0] * n_transitions,
        "llm_call_count": llm_calls[0],
    }


def _run_baseline_planner_no_validation(seed_text: str, k: int) -> dict:
    """Execute the ``baseline_planner_no_validation`` mode.

    For each planned transition, fetches AKG metadata and generates 1-2
    sentences with appraisal-grounded guidance.  No retry on mismatch.

    Parameters
    ----------
    seed_text:
        Opening narrative passage.
    k:
        Total trajectory length.

    Returns
    -------
    dict
        Standardised result dictionary.
    """
    seed_emotion = _detect(seed_text)
    trajectory = _plan(seed_emotion, k)
    llm_calls = [0]

    accumulated_text = seed_text
    segments: list[str] = []
    realized: list[str] = [seed_emotion]
    retry_counts: list[int] = []

    for i in range(len(trajectory) - 1):
        src = trajectory[i]
        tgt = trajectory[i + 1]
        meta = _fetch_meta(src, tgt)

        prompt = _prompt_step(
            seed_text=accumulated_text,
            src_emotion=src,
            tgt_emotion=tgt,
            appraisal_condition=meta["appraisal_condition"],
            behavioral_tendency=meta["behavioral_tendency"],
        )
        segment = _guarded_generate(prompt, llm_calls)
        realized_emotion = _detect(segment)  # logged only; no retry

        accumulated_text = accumulated_text + " " + segment
        segments.append(segment)
        realized.append(realized_emotion)
        retry_counts.append(0)

    return {
        "mode": "baseline_planner_no_validation",
        "seed_emotion": seed_emotion,
        "planned_trajectory": trajectory,
        "realized_emotions": realized,
        "generated_segments": segments,
        "etvs": _compute_etvs(realized),
        "retry_counts": retry_counts,
        "llm_call_count": llm_calls[0],
    }


def _run_full_model(seed_text: str, k: int) -> dict:
    """Execute the ``full_model`` mode.

    For each planned transition, generates a segment, detects the realized
    emotion, and retries up to ``MAX_RETRIES`` times on mismatch.

    Retry policy
    ------------
    On detected mismatch, the prompt is augmented with a reinforcement line
    that makes the target emotion more explicit:

        "The previous attempt did not clearly express {tgt}.
        Make the emotional shift to {tgt} more explicit."

    This augmentation does not alter the planned symbolic trajectory.  After
    ``MAX_RETRIES`` attempts, the most recently generated segment is accepted
    regardless of match status to ensure the story always completes.

    Parameters
    ----------
    seed_text:
        Opening narrative passage.
    k:
        Total trajectory length.

    Returns
    -------
    dict
        Standardised result dictionary.
    """
    seed_emotion = _detect(seed_text)
    trajectory = _plan(seed_emotion, k)
    llm_calls = [0]

    accumulated_text = seed_text
    segments: list[str] = []
    realized: list[str] = [seed_emotion]
    retry_counts: list[int] = []

    for i in range(len(trajectory) - 1):
        src = trajectory[i]
        tgt = trajectory[i + 1]
        meta = _fetch_meta(src, tgt)

        base_prompt = _prompt_step(
            seed_text=accumulated_text,
            src_emotion=src,
            tgt_emotion=tgt,
            appraisal_condition=meta["appraisal_condition"],
            behavioral_tendency=meta["behavioral_tendency"],
        )

        # First attempt uses the base prompt.
        retries = 0
        active_prompt = base_prompt
        segment = _guarded_generate(active_prompt, llm_calls)
        realized_emotion = _detect(segment)

        # Retry loop — capped at MAX_RETRIES, never spins beyond bound.
        while realized_emotion != tgt and retries < MAX_RETRIES:
            retries += 1
            active_prompt = _prompt_step_retry(base_prompt, tgt)
            segment = _guarded_generate(active_prompt, llm_calls)
            realized_emotion = _detect(segment)

        accumulated_text = accumulated_text + " " + segment
        segments.append(segment)
        realized.append(realized_emotion)
        retry_counts.append(retries)

    return {
        "mode": "full_model",
        "seed_emotion": seed_emotion,
        "planned_trajectory": trajectory,
        "realized_emotions": realized,
        "generated_segments": segments,
        "etvs": _compute_etvs(realized),
        "retry_counts": retry_counts,
        "llm_call_count": llm_calls[0],
    }


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

_MODE_DISPATCH: dict = {
    "baseline_free": _run_baseline_free,
    "baseline_sequence_prompt": _run_baseline_sequence_prompt,
    "baseline_planner_no_validation": _run_baseline_planner_no_validation,
    "full_model": _run_full_model,
}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_story(seed_text: str, k: int, mode: str) -> dict:
    """Generate an emotionally grounded story continuation under a given mode.

    Orchestrates the symbolic planner, AKG metadata retrieval, LLM backend,
    emotion detector, and transition validator according to the experimental
    condition specified by *mode*.

    Parameters
    ----------
    seed_text:
        Opening narrative passage (1-3 sentences).  The character's emotional
        state is detected from this text and used as the trajectory start.
    k:
        Total trajectory length, including the seed emotion.  The story
        continuation will contain ``k-1`` generated segments.  Must be >= 2.
    mode:
        Experimental condition identifier.  One of:
        ``"baseline_free"``, ``"baseline_sequence_prompt"``,
        ``"baseline_planner_no_validation"``, ``"full_model"``.

    Returns
    -------
    dict with the following keys:

    ``mode`` : str
        The mode that produced this result.

    ``seed_emotion`` : str
        OCC emotion detected from ``seed_text``.

    ``planned_trajectory`` : list[str]
        Ordered list of ``k`` OCC emotions from the symbolic planner.
        For ``baseline_free``, this is computed for length alignment but not
        used to guide generation.

    ``realized_emotions`` : list[str]
        Ordered list of OCC emotions detected in the seed + each generated
        segment.  Length equals ``k``.

    ``generated_segments`` : list[str]
        Ordered list of ``k-1`` generated story segments.

    ``etvs`` : float
        Emotional Transition Validity Score computed over ``realized_emotions``.

    ``retry_counts`` : list[int]
        Number of retries at each of the ``k-1`` transition steps.  Entries
        are 0 for modes that do not retry.

    ``llm_call_count`` : int
        Total number of ``generate_text`` invocations made by this mode,
        including retries.

    Raises
    ------
    ValueError
        If *mode* is not one of the four valid identifiers, or if *k* < 2.
    TypeError
        If *seed_text* is not a string, or *k* is not an integer.

    Examples
    --------
    ::

        result = generate_story(
            seed_text="She stared at the rejection letter, her hands trembling.",
            k=3,
            mode="full_model",
        )
        print(result["planned_trajectory"])
        print(result["etvs"])
        print(result["llm_call_count"])
    """
    if not isinstance(seed_text, str) or not seed_text.strip():
        raise TypeError("seed_text must be a non-empty string.")
    if not isinstance(k, int) or isinstance(k, bool):
        raise TypeError(f"k must be an integer, got {type(k).__name__!r}.")
    if k < 2:
        raise ValueError(f"k must be >= 2, got {k}.")
    if mode not in _VALID_MODES:
        raise ValueError(
            f"Unknown mode: {mode!r}. "
            f"Valid modes: {sorted(_VALID_MODES)}."
        )

    return _MODE_DISPATCH[mode](seed_text.strip(), k)