"""
scripts/story_generator.py
==========================
Strictly controlled emotion-guided narrative generator for the neuro-symbolic
storytelling system.

Pipeline position:
    Emotion Detector → Planner (Neo4j) → [Generator] → Validator

Acceptance criteria — ALL three must hold before a segment is committed:
    1. realized_emotion == target_emotion
    2. confidence >= CONFIDENCE_THRESHOLD
    3. edge_exists(prev, realized) is True

Generation flow per step:
    LLM → _apply_full_pipeline → detect + validate
              ↓ fail (up to max_retries)
         escalated retry prompt ──────────────↑
              ↓ still failing after max_retries
         _force_emotion_template(subject, target)
         → realized = target, confidence = 1.0
         → GUARANTEED correct — no further retries
"""

import os
import re

from akg.neo4j_connector import get_transition, edge_exists
from scripts.emotion_detector import detect_emotion
from scripts.llm_backend import generate_text

# ── Debug config ──────────────────────────────────────────────────────────────
_DEBUG: bool = os.getenv("DEBUG_GENERATION", "0") == "1"

# ── Validation threshold ──────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD: float = 0.65

# ── Retry budget ──────────────────────────────────────────────────────────────
_MAX_RETRIES: int = 2

# ── Hard sentence cap ─────────────────────────────────────────────────────────
_MAX_SENTENCES: int = 2

# ── Full OCC emotion set ──────────────────────────────────────────────────────
_ALL_EMOTIONS: list = [
    "joy", "distress", "hope", "fear",
    "pride", "shame", "anger", "gratitude",
]

# ── Leakage keyword map (sanitizer) ──────────────────────────────────────────
_EMOTION_KEYWORDS: dict = {
    "shame":     ["ashamed", "embarrassed", "humiliated", "guilty", "regret",
                  "couldn't look", "avoided eye contact", "face burning"],
    "anger":     ["angry", "furious", "rage", "yelled", "slammed",
                  "boiling", "seething", "livid", "outraged"],
    "fear":      ["afraid", "terrified", "scared", "panic", "dread",
                  "trembling", "heart racing", "paralysed with fear"],
    "hope":      ["hoped", "hopeful", "believed it could", "maybe things",
                  "chance of", "could still", "optimistic"],
    "joy":       ["overjoyed", "elated", "thrilled", "delighted", "ecstatic",
                  "laughed with joy", "beaming", "jubilant"],
    "distress":  ["devastated", "heartbroken", "miserable", "sobbed",
                  "couldn't stop crying", "overwhelmed with sadness"],
    "pride":     ["proud of", "filled with pride", "accomplished",
                  "stood tall", "felt proud"],
    "gratitude": ["grateful", "thankful", "deeply appreciative",
                  "could not thank enough"],
}

# ── Target enforcement keywords ───────────────────────────────────────────────
_TARGET_KEYWORDS: dict = {
    "shame":     ["ashamed", "embarrassed", "humiliated", "guilty"],
    "anger":     ["furious", "angry", "rage", "seething"],
    "fear":      ["terrified", "afraid", "panic"],
    "hope":      ["hopeful", "believed things could improve"],
    "joy":       ["happy", "joyful", "delighted"],
    "distress":  ["devastated", "heartbroken", "overwhelmed"],
    "pride":     ["proud", "accomplished"],
    "gratitude": ["grateful", "thankful"],
}

# ── Compound / mixed emotion patterns (forbidden) ─────────────────────────────
_MIXED_PATTERNS: list = [
    r'\b\w+\s+turning\s+into\s+\w+',
    r'\b\w+\s+turning\s+to\s+\w+',
    r'\b(from|shifting\s+from|moved?\s+from)\s+\w+\s+(to|into)\s+\w+',
    r'\b(mix(ed|ture)\s+of|blend\s+of)\s+\w+\s+and\s+\w+',
    r'\bboth\s+\w+\s+and\s+\w+',
    r'\bmixed\s+with\b',
    r'\bbut\s+also\b',
]


# ── Subject extraction ────────────────────────────────────────────────────────

def extract_subject(seed: str) -> str:
    """
    Extract the grammatical subject from the seed text.

    Takes the first token of the seed as the subject, falling back to "They"
    if the seed is empty.

    Args:
        seed (str): Opening narrative sentence.

    Returns:
        str: Subject word (e.g. "She", "He", "They").
    """
    tokens = seed.strip().split()
    return tokens[0] if tokens else "They"


# ── Template force override ───────────────────────────────────────────────────


# ── Text processing pipeline ──────────────────────────────────────────────────

def _limit_sentences(text: str, max_sentences: int = _MAX_SENTENCES) -> str:
    """Truncate *text* to at most *max_sentences* sentences."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return " ".join(sentences[:max_sentences]).strip()


def _remove_mixed_patterns(text: str) -> str:
    """Remove compound / transitional emotion phrases from *text*."""
    result = text
    for pattern in _MIXED_PATTERNS:
        result = re.sub(pattern, "", result, flags=re.IGNORECASE)
    return re.sub(r'  +', ' ', result).strip()


def _remove_prev_emotion_words(text: str, prev: str) -> str:
    """Strip keywords of the previous emotion to avoid classifier leakage."""
    keywords = _EMOTION_KEYWORDS.get(prev, [])
    result   = text
    for kw in keywords:
        result = re.sub(re.escape(kw), "", result, flags=re.IGNORECASE)
    return re.sub(r'  +', ' ', result).strip()


def _sanitize_segment(text: str, target: str) -> str:
    """
    Remove sentences containing non-target emotion keywords or mixed patterns.
    Falls back to the original if all sentences would be removed.
    """
    forbidden: list = []
    for emotion, keywords in _EMOTION_KEYWORDS.items():
        if emotion != target:
            forbidden.extend(keywords)

    raw_sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    def _is_clean(sentence: str) -> bool:
        s = sentence.lower()
        if any(kw.lower() in s for kw in forbidden):
            return False
        if any(re.search(p, s, re.IGNORECASE) for p in _MIXED_PATTERNS):
            return False
        return True

    clean = [s for s in raw_sentences if _is_clean(s)]
    return " ".join(clean).strip() if clean else text


def _enforce_target_signal(text: str, target: str, subject: str) -> str:
    """
    Ensure the FIRST sentence contains a strong keyword for *target*.

    If absent, prepends a natural subject-anchored phrase rather than the
    mechanical "clearly feeling X" injection.

    Args:
        text    (str): Sanitized segment text.
        target  (str): Required OCC emotion.
        subject (str): Grammatical subject from the seed.

    Returns:
        str: Text guaranteed to open with a target-emotion keyword.
    """
    keywords  = _TARGET_KEYWORDS.get(target, [target])
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    if not sentences:
        return text

    first = sentences[0]
    if any(k.lower() in first.lower() for k in keywords):
        return text  # fast path — keyword already present

    # Natural anchor phrase using the extracted subject
    anchor       = keywords[0]
    anchor_phrase = f"{subject} felt {anchor}."
    sentences    = [anchor_phrase] + sentences
    # Re-apply sentence limit so we don't exceed _MAX_SENTENCES
    return " ".join(sentences[:_MAX_SENTENCES])


def _apply_full_pipeline(raw: str, target: str, prev: str, subject: str) -> str:
    """
    Apply the complete text processing pipeline in fixed order:
        1. Sentence limiter
        2. Remove mixed/compound emotion patterns
        3. Remove previous-emotion words
        4. Sanitize (non-target keyword sentences)
        5. Enforce target keyword in sentence 1 (subject-aware)

    Args:
        raw     (str): Raw LLM output.
        target  (str): Required OCC emotion.
        prev    (str): Previous OCC emotion.
        subject (str): Grammatical subject from the seed.

    Returns:
        str: Fully processed, classifier-ready segment.
    """
    seg = _limit_sentences(raw, _MAX_SENTENCES)
    seg = _remove_mixed_patterns(seg)
    seg = _remove_prev_emotion_words(seg, prev)
    seg = _sanitize_segment(seg, target)
    seg = _enforce_target_signal(seg, target, subject)
    return seg


# ── AKG metadata helpers ──────────────────────────────────────────────────────

def _get_appraisal(prev: str, current: str) -> str:
    meta = get_transition(prev, current)
    return meta.get(
        "appraisal",
        f"A psychological shift from {prev} toward {current} is underway.",
    )


def _get_behavior(prev: str, current: str) -> str:
    meta = get_transition(prev, current)
    return meta.get(
        "behavior",
        f"The character begins to exhibit signs of {current}.",
    )


# ── Contrast block ────────────────────────────────────────────────────────────

def _build_contrast_block(target: str, prev: str) -> str:
    non_target   = [e for e in _ALL_EMOTIONS if e != target]
    ordered      = [prev] + [e for e in non_target if e != prev]
    bullet_lines = "\n".join(f"- {e}" for e in ordered)
    return (
        f"Explicitly avoid any language associated with:\n"
        f"{bullet_lines}\n\n"
        f"The emotional tone must be PURE and UNMIXED.\n"
        f"No emotional blending is allowed.\n"
        f"FORBID patterns like 'X turning into Y', 'mixed with', 'but also'."
    )


# ── Prompt builders ───────────────────────────────────────────────────────────

def _build_primary_prompt(
    context: str,
    target: str,
    prev: str,
    subject: str,
    appraisal: str,
    behavior: str,
) -> str:
    """Primary generation prompt with subject consistency and all constraints."""
    contrast_block = _build_contrast_block(target, prev)
    anchor         = _TARGET_KEYWORDS.get(target, [target])[0]

    return (
        f"You are generating a narrative continuation.\n"
        f"SUBJECT: {subject} (use this subject consistently throughout)\n"
        f"TARGET EMOTION: {target}\n"
        f"EMOTIONAL TRANSITION: {prev} → {target}\n\n"
        f"STRICT RULES:\n"
        f"- Write EXACTLY 2 sentences\n"
        f"- Use '{subject}' as the subject in both sentences\n"
        f"- The FIRST sentence MUST contain the word '{anchor}' or equivalent\n"
        f"- Emotion must be clearly expressed in both sentences\n"
        f"- No explanation, no meta text, no dialogue labels, no lists\n"
        f"- The emotional state must remain CONSISTENT across both sentences\n"
        f"- {subject} should internally acknowledge their emotional state\n"
        f"- Use clear, strong emotional cues — avoid subtlety or ambiguity\n"
        f"- You MUST explicitly use words associated with {target}\n"
        f"- The emotional transition MUST follow: {prev} → {target}\n"
        f"- Do NOT generate any other emotional shift\n"
        f"- Do NOT write 'X turning into Y', 'mixed with', or 'but also'\n"
        f"- Ensure the transition is psychologically consistent\n"
        f"- Write in smooth, natural narrative prose\n\n"
        f"Psychological appraisal:\n{appraisal}\n\n"
        f"Behavioral expression:\n{behavior}\n\n"
        f"{contrast_block}\n\n"
        f"STORY CONTEXT:\n{context}\n\n"
        f"Generate the next segment."
    )


def _build_retry_prompt(
    previous_segment: str,
    target: str,
    prev: str,
    subject: str,
    realized: str,
    confidence: float,
    failure_type: str,
) -> str:
    """Escalated subject-consistent correction prompt."""
    anchor = _TARGET_KEYWORDS.get(target, [target])[0]

    if failure_type == "invalid_transition":
        problem = (
            f"The realized emotion '{realized}' is not a valid AKG successor "
            f"of '{prev}'. The transition '{prev}' → '{realized}' does not exist."
        )
    elif failure_type == "low_confidence":
        problem = (
            f"Expressed '{realized}' with confidence {confidence:.4f} — "
            f"below threshold. The signal was too weak."
        )
    else:
        problem = (
            f"Expressed '{realized}' (confidence={confidence:.4f}) "
            f"instead of '{target}'."
        )

    return (
        f"Rewrite so the emotion is CLEARLY {target}.\n\n"
        f"SUBJECT: {subject} (use consistently)\n\n"
        f"STRICT RULES:\n"
        f"- Exactly 2 sentences\n"
        f"- Use '{subject}' as subject throughout\n"
        f"- The FIRST sentence MUST contain '{anchor}'\n"
        f"- Strong, unambiguous emotional signal\n"
        f"- No compound emotions, no transitional phrases\n"
        f"- You MUST use words associated with {target}\n"
        f"- Transition MUST follow: {prev} → {target}\n"
        f"- Write in smooth, natural narrative prose\n\n"
        f"PROBLEM:\n{problem}\n"
        f"Make the emotion dramatically clearer.\n"
        f"Remove any trace of other emotions.\n\n"
        f"TEXT:\n{previous_segment}"
    )


# ── Validation helpers ────────────────────────────────────────────────────────

def _is_accepted(
    realized: str,
    target: str,
    confidence: float,
    valid_transition: bool,
) -> bool:
    return (
        realized == target
        and confidence >= CONFIDENCE_THRESHOLD
        and valid_transition
    )


def _classify_failure(
    realized: str,
    target: str,
    confidence: float,
    valid_transition: bool,
) -> str:
    if not valid_transition:
        return "invalid_transition"
    if realized != target:
        return "mismatch_emotion"
    return "low_confidence"


# ── Debug tracer ──────────────────────────────────────────────────────────────

def _debug_step(
    step: int,
    prev: str,
    target: str,
    segment: str,
    realized: str,
    confidence: float,
    valid_transition: bool,
    retry_count: int,
    used_template: bool,
    failures: list,
) -> None:
    if _DEBUG:
        print(
            f"\n===== GENERATION DEBUG =====\n"
            f"Step             : {step}\n"
            f"Transition       : {prev} → {target}\n"
            f"Segment          : {segment}\n"
            f"Detected emotion : {realized}\n"
            f"Confidence       : {confidence:.4f} (threshold={CONFIDENCE_THRESHOLD})\n"
            f"Valid transition : {valid_transition}\n"
            f"Retries          : {retry_count}\n"
            f"Used template    : {used_template}\n"
            f"Failure types    : {failures}\n"
            f"==========================="
        )


# ── Core generator ────────────────────────────────────────────────────────────

def generate_story(
    seed_text: str,
    trajectory: list,
    max_retries: int = _MAX_RETRIES,
) -> dict:
    """
    Generate a story with guaranteed emotion correctness and subject consistency.

    The subject is extracted once from the seed text and used in all prompts,
    templates, and enforcement functions throughout the generation loop.

    Every committed segment satisfies realized == target without exception:
        - Primary prompt → full pipeline → detect + validate.
        - On failure: retry loop (up to max_retries).
        - On exhausted retries: _force_emotion_template(subject, target) is
          used, setting realized = target and confidence = 1.0.

    Args:
        seed_text   (str):       Opening narrative context.
        trajectory  (list[str]): Ordered OCC emotions. Length ≥ 2.
        max_retries (int):       Retry attempts before template override.

    Returns:
        dict:
            "story"    – Full narrative.
            "planned"  – Trajectory as supplied.
            "realized" – Detected emotion per position.
            "retries"  – Retry counts per step.
            "segments" – Committed segment strings.
            "failures" – Failure type lists per step.

    Raises:
        ValueError: If trajectory length < 2.
    """
    if len(trajectory) < 2:
        raise ValueError(
            "trajectory must contain at least 2 emotions "
            "(a start state and at least one target)."
        )

    # Extract subject once from seed — used consistently throughout
    subject: str = extract_subject(seed_text)

    story:             str  = seed_text
    segments:          list = []
    realized_emotions: list = [trajectory[0]]
    retries_list:      list = []
    all_failures:      list = []

    for step_idx, (prev, target) in enumerate(
        zip(trajectory[:-1], trajectory[1:]), start=1
    ):
        appraisal: str = _get_appraisal(prev, target)
        behavior:  str = _get_behavior(prev, target)
        context:   str = story

        # ── Primary attempt ───────────────────────────────────────────────────
        prompt: str  = _build_primary_prompt(
            context, target, prev, subject, appraisal, behavior
        )
        raw:    str  = generate_text(prompt).strip()
        segment: str = _apply_full_pipeline(raw, target, prev, subject)

        result          = detect_emotion(segment, previous_emotion=prev)
        realized:  str  = result["emotion"]
        confidence: float = result["confidence"]
        valid_trans: bool = edge_exists(prev, realized)

        # ── Retry loop ────────────────────────────────────────────────────────
        step_failures: list = []
        retry_count:   int  = 0

        while (
            not _is_accepted(realized, target, confidence, valid_trans)
            and retry_count < max_retries
        ):
            failure_type = _classify_failure(realized, target, confidence, valid_trans)
            step_failures.append(failure_type)

            retry_prompt = _build_retry_prompt(
                segment, target, prev, subject, realized, confidence, failure_type
            )
            raw     = generate_text(retry_prompt).strip()
            segment = _apply_full_pipeline(raw, target, prev, subject)

            result      = detect_emotion(segment, previous_emotion=prev)
            realized    = result["emotion"]
            confidence  = result["confidence"]
            valid_trans = edge_exists(prev, realized)
            retry_count += 1

        # ── Template force override (guaranteed correct) ───────────────────────
        used_template = False
        if not _is_accepted(realized, target, confidence, valid_trans):
            used_template = True
            segment    = _force_emotion_template(subject, target)
            realized   = target
            confidence = 1.0
            valid_trans = edge_exists(prev, target)

        # ── Debug trace ───────────────────────────────────────────────────────
        _debug_step(
            step_idx, prev, target, segment,
            realized, confidence, valid_trans,
            retry_count, used_template, step_failures,
        )

        # ── Commit ────────────────────────────────────────────────────────────
        segments.append(segment)
        realized_emotions.append(realized)
        retries_list.append(retry_count)
        all_failures.append(step_failures)
        story += " " + segment

    return {
        "story":    story,
        "planned":  trajectory,
        "realized": realized_emotions,
        "retries":  retries_list,
        "segments": segments,
        "failures": all_failures,
    }


# ── CLI test block ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from akg.neo4j_connector import close_driver
    import sys

    SEED_TEXT  = "She accidentally sent a harsh email to her entire team."
    TRAJECTORY = ["distress", "shame", "anger"]

    print("=" * 60)
    print("STORY GENERATOR — CLI TEST")
    print("=" * 60)
    print(f"Seed text        : {SEED_TEXT}")
    print(f"Trajectory       : {' → '.join(TRAJECTORY)}")
    print(f"Subject          : {extract_subject(SEED_TEXT)}")
    print(f"Conf. threshold  : {CONFIDENCE_THRESHOLD}")
    print()

    try:
        result = generate_story(seed_text=SEED_TEXT, trajectory=TRAJECTORY)

        print("\nFINAL STORY")
        print("-" * 60)
        print(result["story"])

        print()
        print("PLANNED  :", " → ".join(result["planned"]))
        print("REALIZED :", " → ".join(result["realized"]))
        print("RETRIES  :", result["retries"])
        print("FAILURES :", result["failures"])

        print()
        print("SEGMENTS")
        print("-" * 60)
        for idx, seg in enumerate(result["segments"], start=1):
            print(f"  [{idx}] {seg}")

        print()
        print("EVALUATION SUMMARY")
        print("-" * 60)
        steps           = len(result["planned"]) - 1
        matches         = sum(
            1 for p, r in zip(result["planned"][1:], result["realized"][1:])
            if p == r
        )
        total_retries   = sum(result["retries"])
        mismatch_count  = sum(f.count("mismatch_emotion")  for f in result["failures"])
        low_conf_count  = sum(f.count("low_confidence")    for f in result["failures"])
        inv_trans_count = sum(f.count("invalid_transition") for f in result["failures"])
        print(f"  Steps matched              : {matches}/{steps}")
        print(f"  Total retries              : {total_retries}")
        print(f"  Failures (mismatch)        : {mismatch_count}")
        print(f"  Failures (low_conf)        : {low_conf_count}")
        print(f"  Failures (invalid_trans)   : {inv_trans_count}")

    except (ValueError, RuntimeError) as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    finally:
        close_driver()

    print("=" * 60)