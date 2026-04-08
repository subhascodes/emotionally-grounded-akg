"""
scripts/experiment_runner.py
============================
Deterministic evaluation framework for the neuro-symbolic storytelling system.

Ablation study — four systems:
    1. baseline_free    — unconstrained single LLM call, split into segments
    2. baseline_prompt  — context-shift prompt, no AKG, no retry
    3. planner_only     — trajectory used, appraisal-only prompt, no metadata,
                          no retry, no sanitizer
    4. full_model       — complete pipeline (AKG + appraisal + retry + sanitizer)

No explicit emotion labels appear in any prompt sent to the LLM.

Metrics (per sample):
    ETVS          — Emotion Trajectory Validity Score
    Accuracy      — pure realized == planned match rate
    Soft Accuracy — partial credit for graph-adjacent mismatches
    CSR           — Constraint Satisfaction Rate (realized transitions on graph)
    Retries       — average retries per step (full_model only)
"""

import csv
import json
import math
import os
import sys

from scripts.story_generator import generate_story
from scripts.emotion_detector import detect_emotion
from scripts.emotion_planner import plan_emotion_trajectory
from scripts.llm_backend import generate_text
from scripts.trajectory_explainer import explain_trajectory, print_explanation
from scripts.graph_visualizer import visualize_trajectory
from akg.neo4j_connector import edge_exists, close_driver, get_transition

# ── Global config ─────────────────────────────────────────────────────────────
DATA_PATH   = "data/rocstories_subset.csv"
MAX_SAMPLES = 1
N_SAMPLES   = 50
K           = 3


# ── Dataset loader ────────────────────────────────────────────────────────────

def load_rocstories_subset(path: str) -> list:
    """
    Load seed texts from a ROCStories CSV file.
    Expected columns: sentence1, sentence2.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. "
            f"Ensure data/rocstories_subset.csv is present."
        )

    seeds = []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if "sentence1" not in (reader.fieldnames or []) or \
           "sentence2" not in (reader.fieldnames or []):
            raise ValueError(
                "CSV must contain columns 'sentence1' and 'sentence2'."
            )
        for row in reader:
            s1 = row["sentence1"].strip()
            s2 = row["sentence2"].strip()
            combined = (s1 + " " + s2).strip()
            if combined:
                seeds.append(combined)

    return seeds


# ── Statistics helper ─────────────────────────────────────────────────────────

def mean_std(values: list) -> tuple:
    """Return (mean, population_std). Returns (0.0, 0.0) if empty."""
    if not values:
        return 0.0, 0.0
    n   = len(values)
    mu  = sum(values) / n
    var = sum((v - mu) ** 2 for v in values) / n
    return mu, math.sqrt(var)


# ── Soft accuracy helper ──────────────────────────────────────────────────────

def _soft_score(realized: str, target: str) -> float:
    """
    1.0 exact match, 0.5 graph-adjacent, 0.0 otherwise.
    """
    if realized == target:
        return 1.0
    elif edge_exists(realized, target):
        return 0.5
    return 0.0


# ── Metric computation ────────────────────────────────────────────────────────

def _compute_metrics(
    planned:  list,
    realized: list,
    retries:  list,
    failures: list,
) -> dict:
    """
    Compute ETVS, Accuracy, Soft Accuracy, CSR, and avg retries.
    All metrics operate on the realized trajectory.
    """
    steps = len(planned) - 1
    if steps <= 0:
        return {
            "etvs": 0.0, "accuracy": 0.0, "soft_accuracy": 0.0,
            "csr": 0.0, "retries": 0.0,
        }

    etvs_matches = 0
    accurate_steps = 0
    soft_total = 0.0

    for i in range(1, len(planned)):
        correct_emotion  = (
            realized[i] == planned[i]
            if i < len(realized) else False
        )
        valid_transition = edge_exists(
            realized[i - 1] if (i - 1) < len(realized) else "",
            realized[i]     if i < len(realized) else "",
        )
        if correct_emotion and valid_transition:
            etvs_matches += 1
        if correct_emotion:
            accurate_steps += 1
        soft_total += _soft_score(
            realized[i] if i < len(realized) else "",
            planned[i],
        )

    total_edges = len(realized) - 1
    valid_edges = 0
    for i in range(total_edges):
        if edge_exists(realized[i], realized[i + 1]):
            valid_edges += 1
    csr = valid_edges / total_edges if total_edges > 0 else 0.0

    return {
        "etvs":          etvs_matches / steps,
        "accuracy":      accurate_steps / steps,
        "soft_accuracy": soft_total / steps,
        "csr":           csr,
        "retries":       (sum(retries) / steps) if retries else 0.0,
    }


# ── AKG metadata helper for baselines ────────────────────────────────────────

def _get_transition_meta(prev: str, target: str) -> tuple:
    """
    Fetch appraisal and behavior from Neo4j for use in baseline prompts.
    Returns generic strings if the edge has no metadata.
    """
    meta = get_transition(prev, target)
    appraisal = meta.get(
        "appraisal",
        "A shift in internal evaluation is occurring.",
    )
    behavior = meta.get(
        "behavior",
        "The character exhibits a change in behavior and internal state.",
    )
    return appraisal, behavior


# ── System implementations ────────────────────────────────────────────────────

def _run_baseline_free(seed_text: str, trajectory: list) -> tuple:
    """
    System 1 — Baseline Free.
    Single unconstrained LLM call with no emotion guidance.
    Returns: (metrics, realized, retries, story)
    """
    import re

    subject = seed_text.strip().split()[0] if seed_text.strip() else "They"

    prompt = (
        f"Continue this story naturally for {K} more sentences.\n\n"
        f"Story so far:\n{seed_text}\n\n"
        f"Constraints:\n"
        f"- Do NOT explicitly name emotions\n"
        f"- Maintain consistent subject identity (no gender or name changes)\n"
        f"- Use '{subject}' as the subject throughout\n"
        f"- Write in smooth, natural narrative prose"
    )
    try:
        raw = generate_text(prompt).strip()
    except Exception:
        raw = seed_text

    story     = raw
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', raw) if s.strip()]
    segments  = sentences[:K] if len(sentences) >= K else sentences

    while len(segments) < K:
        segments.append(seed_text)

    realized = [trajectory[0]]
    prev     = trajectory[0]
    for seg in segments[: len(trajectory) - 1]:
        try:
            res = detect_emotion(seg, previous_emotion=prev)
            realized.append(res["emotion"])
            prev = res["emotion"]
        except Exception:
            realized.append(prev)

    metrics = _compute_metrics(trajectory, realized, [], [])
    return metrics, realized, [], story


def _run_baseline_prompt(seed_text: str, trajectory: list) -> tuple:
    """
    System 2 — Baseline Prompt.
    Per-step prompt uses appraisal/behavior context cues only.
    No AKG edge validation, no retry, no sanitizer.
    Returns: (metrics, realized, retries, story)
    """
    subject  = seed_text.strip().split()[0] if seed_text.strip() else "They"
    story    = seed_text
    realized = [trajectory[0]]
    prev     = trajectory[0]

    for target in trajectory[1:]:
        appraisal, behavior = _get_transition_meta(prev, target)

        prompt = (
            f"Continue the story to reflect a psychological shift consistent "
            f"with the context.\n\n"
            f"Story so far:\n{story}\n\n"
            f"Psychological appraisal:\n{appraisal}\n\n"
            f"Behavioral tendency:\n{behavior}\n\n"
            f"Constraints:\n"
            f"- Do NOT explicitly name emotions\n"
            f"- Express the internal state through actions, thoughts, and tone\n"
            f"- Maintain consistent subject identity (no gender or name changes)\n"
            f"- Use '{subject}' as the subject throughout\n"
            f"- Write exactly 2 sentences"
        )
        try:
            segment = generate_text(prompt).strip()
        except Exception:
            segment = ""

        try:
            res = detect_emotion(segment, previous_emotion=prev)
            emo = res["emotion"]
        except Exception:
            emo = prev

        realized.append(emo)
        story += (" " + segment) if segment else ""
        prev   = emo

    metrics = _compute_metrics(trajectory, realized, [], [])
    return metrics, realized, [], story


def _run_planner_only(seed_text: str, trajectory: list) -> tuple:
    """
    System 3 — Planner Only.
    Trajectory used; prompt includes appraisal/behavior from AKG only.
    No contrast block, no retry, no sanitizer.
    Returns: (metrics, realized, retries, story)
    """
    subject  = seed_text.strip().split()[0] if seed_text.strip() else "They"
    story    = seed_text
    realized = [trajectory[0]]
    prev     = trajectory[0]

    for target in trajectory[1:]:
        appraisal, behavior = _get_transition_meta(prev, target)

        prompt = (
            f"Continue the narrative naturally.\n\n"
            f"Story so far:\n{story}\n\n"
            f"Psychological appraisal:\n{appraisal}\n\n"
            f"Behavioral tendency:\n{behavior}\n\n"
            f"Constraints:\n"
            f"- Do NOT explicitly name emotions\n"
            f"- Maintain consistent subject identity (no gender or name changes)\n"
            f"- Use '{subject}' as the subject throughout\n"
            f"- Write exactly 2 sentences"
        )
        try:
            segment = generate_text(prompt).strip()
        except Exception:
            segment = ""

        try:
            res = detect_emotion(segment, previous_emotion=prev)
            emo = res["emotion"]
        except Exception:
            emo = prev

        realized.append(emo)
        story += (" " + segment) if segment else ""
        prev   = emo

    metrics = _compute_metrics(trajectory, realized, [], [])
    return metrics, realized, [], story


def _run_full_model(seed_text: str, trajectory: list) -> tuple:
    """
    System 4 — Full Model.
    Complete pipeline: AKG appraisal+behavior + sanitizer + retry.
    No explicit emotion labels in any prompt.
    Returns: (metrics, realized, retries, story)
    """
    try:
        result   = generate_story(
            seed_text=seed_text,
            trajectory=trajectory,
        )
        story    = result["story"]
        realized = result["realized"]
        retries  = result["retries"]
        failures = result.get("failures", [])
        metrics  = _compute_metrics(result["planned"], realized, retries, failures)
        return metrics, realized, retries, story
    except Exception as exc:
        sys.stderr.write(f"[full_model] generate_story failed: {exc}\n")
        return (
            {"etvs": 0.0, "accuracy": 0.0, "soft_accuracy": 0.0,
             "csr": 0.0, "retries": 0.0},
            [trajectory[0]],
            [],
            "N/A",
        )


# ── Main evaluation loop ──────────────────────────────────────────────────────

def run_experiment() -> None:
    """
    Run the full ablation evaluation and print clean, paper-ready output.
    """
    seeds = load_rocstories_subset(DATA_PATH)
    seeds = seeds[:MAX_SAMPLES]

    if not seeds:
        raise ValueError("No valid seed texts found in the dataset.")

    all_results: dict = {
        "baseline_free":   [],
        "baseline_prompt": [],
        "planner_only":    [],
        "full_model":      [],
    }

    runners = [
        ("baseline_free",   _run_baseline_free),
        ("baseline_prompt", _run_baseline_prompt),
        ("planner_only",    _run_planner_only),
        ("full_model",      _run_full_model),
    ]

    for i, seed_text in enumerate(seeds):

        # ── Shared trajectory ─────────────────────────────────────────────────
        try:
            start_emotion = detect_emotion(seed_text)["emotion"]
        except Exception:
            start_emotion = "distress"

        try:
            trajectory = plan_emotion_trajectory(
                start_emotion=start_emotion,
                k=K,
                seed=i,
            )
        except Exception as exc:
            sys.stderr.write(f"[planner] failed for sample {i}: {exc}\n")
            trajectory = [start_emotion] * K

        # ── Sample header ─────────────────────────────────────────────────────
        print(f"""
============================================================
SAMPLE {i + 1} — ABLATION ANALYSIS
============================================================
SEED:
{seed_text}
PLANNED TRAJECTORY:
{" → ".join(trajectory)}
""")

        # ── Run all systems ───────────────────────────────────────────────────
        sample_results: dict = {}

        for system_name, runner in runners:
            try:
                metrics, realized, retries, story = runner(seed_text, trajectory)
            except Exception as exc:
                sys.stderr.write(f"[{system_name}] error on sample {i}: {exc}\n")
                metrics  = {
                    "etvs": 0.0, "accuracy": 0.0, "soft_accuracy": 0.0,
                    "csr": 0.0, "retries": 0.0,
                }
                realized = [start_emotion]
                retries  = []
                story    = "N/A"

            sample_results[system_name] = {
                "story":    story,
                "realized": realized,
                "metrics":  metrics,
            }
            all_results[system_name].append(metrics)

        # ── Per-system clean output ───────────────────────────────────────────
        for name, data in sample_results.items():
            m = data["metrics"]
            print(f"""------------------------------------------------------------
{name.upper()}
------------------------------------------------------------
Story:
{data['story']}
Realized:
{" → ".join(data['realized'])}
ETVS     : {m['etvs']:.2f}
Accuracy : {m['accuracy']:.2f}
SoftAcc  : {m['soft_accuracy']:.2f}
CSR      : {m['csr']:.2f}
Retries  : {m['retries']:.2f}
""")

        # ── Per-sample ablation table ─────────────────────────────────────────
        print("------------------------------------------------------------")
        print("ABLATION TABLE")
        print("------------------------------------------------------------")
        print(f"{'System':<17}{'ETVS':<8}{'Acc':<8}{'Soft':<8}{'CSR':<8}{'Retry':<8}")
        print("------------------------------------------------------------")
        for name, data in sample_results.items():
            m = data["metrics"]
            print(
                f"{name:<17}"
                f"{m['etvs']:.2f}    "
                f"{m['accuracy']:.2f}    "
                f"{m['soft_accuracy']:.2f}    "
                f"{m['csr']:.2f}    "
                f"{m['retries']:.2f}"
            )
        print("============================================================")

        # ── Traceability block ────────────────────────────────────────────────
        full_model_realized = sample_results.get("full_model", {}).get(
            "realized", trajectory
        )
        try:
            explanation, cypher_query = explain_trajectory(full_model_realized)
            print_explanation(explanation, cypher_query)
        except Exception as exc:
            sys.stderr.write(f"[traceability] failed: {exc}\n")

        # ── Graph visualization ───────────────────────────────────────────────
        try:
            graph_path = f"outputs/sample_{i + 1}_graph.png"
            visualize_trajectory(full_model_realized, graph_path)
        except Exception as exc:
            sys.stderr.write(f"[graph_visualizer] failed for sample {i + 1}: {exc}\n")

    # ── Save raw results ──────────────────────────────────────────────────────
    with open("experiment_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # ── Final aggregated table (multi-sample only) ────────────────────────────
    if len(seeds) > 1:
        aggregated = {
            system: {
                metric: mean_std([e[metric] for e in entries])
                for metric in ("etvs", "accuracy", "soft_accuracy", "csr", "retries")
            }
            for system, entries in all_results.items()
        }

        def fmt(mu_std: tuple) -> str:
            mu, sd = mu_std
            return f"{mu:.2f}±{sd:.2f}"

        col_w = 13
        print()
        print("=" * 68)
        print("FINAL RESULTS (MEAN ± STD)")
        print("=" * 68)
        print()
        print(
            f"{'System':<17}"
            f"{'ETVS':<{col_w}}"
            f"{'Accuracy':<{col_w}}"
            f"{'SoftAcc':<{col_w}}"
            f"{'CSR':<{col_w}}"
            f"{'Retries':<{col_w}}"
        )
        print("-" * 68)

        rows = [
            ("Baseline Free",  "baseline_free",   False),
            ("Prompt",         "baseline_prompt",  False),
            ("Planner Only",   "planner_only",     False),
            ("Full Model",     "full_model",        True),
        ]

        for label, key, show_retries in rows:
            agg          = aggregated[key]
            retries_cell = fmt(agg["retries"]) if show_retries else "0.00"
            print(
                f"{label:<17}"
                f"{fmt(agg['etvs']):<{col_w}}"
                f"{fmt(agg['accuracy']):<{col_w}}"
                f"{fmt(agg['soft_accuracy']):<{col_w}}"
                f"{fmt(agg['csr']):<{col_w}}"
                f"{retries_cell:<{col_w}}"
            )
        print()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        run_experiment()
    except (FileNotFoundError, ValueError) as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        close_driver()