"""
scripts/aggregate_results.py
=============================
Final evaluation aggregation script for the AKG framework.

Runs all four ablation systems over N ROCStories seeds, computes mean
metrics across all samples, and prints a clean paper-ready table.

Usage:
    python scripts/aggregate_results.py
"""

import os
import sys

from scripts.experiment_runner import (
    _run_baseline_free,
    _run_baseline_prompt,
    _run_planner_only,
    _run_full_model,
    load_rocstories_subset,
)
from scripts.emotion_planner import plan_emotion_trajectory
from scripts.emotion_detector import detect_emotion
from akg.neo4j_connector import close_driver

# ── Config ────────────────────────────────────────────────────────────────────
CSV_PATH = "data/rocstories_subset.csv"
K        = 3
N        = 50
OUTPUT_PATH = "outputs/final_results.txt"

# ── Runners ───────────────────────────────────────────────────────────────────
RUNNERS = [
    ("baseline_free",   _run_baseline_free),
    ("baseline_prompt", _run_baseline_prompt),
    ("planner_only",    _run_planner_only),
    ("full_model",      _run_full_model),
]

METRICS = ("etvs", "accuracy", "soft_accuracy", "csr", "retries")

ZERO_METRICS = {m: 0.0 for m in METRICS}


# ── Aggregation helper ────────────────────────────────────────────────────────

def _mean(values: list) -> float:
    """Return mean of *values*. Returns 0.0 for empty lists."""
    return sum(values) / len(values) if values else 0.0


def _aggregate(entries: list) -> dict:
    """Compute per-metric mean across all sample entries."""
    return {
        metric: _mean([e[metric] for e in entries])
        for metric in METRICS
    }


# ── Table formatter ───────────────────────────────────────────────────────────

def _format_table(aggregated: dict) -> str:
    """Return the final paper-ready results table as a string."""
    lines = []
    sep   = "=" * 60
    dash  = "-" * 60

    lines.append(sep)
    lines.append("FINAL AGGREGATED RESULTS (AKG EVALUATION)")
    lines.append(sep)
    lines.append(
        f"{'System':<17}{'ETVS':<8}{'Accuracy':<12}{'SoftAcc':<11}{'CSR':<8}{'Retries':<8}"
    )
    lines.append(dash)

    display_names = {
        "baseline_free":   "baseline_free",
        "baseline_prompt": "baseline_prompt",
        "planner_only":    "planner_only",
        "full_model":      "full_model",
    }
    show_retries = {"full_model"}

    for system, _ in RUNNERS:
        m    = aggregated[system]
        name = display_names[system]
        ret  = f"{m['retries']:.2f}" if system in show_retries else "0.00"
        lines.append(
            f"{name:<17}"
            f"{m['etvs']:.2f}    "
            f"{m['accuracy']:.2f}        "
            f"{m['soft_accuracy']:.2f}       "
            f"{m['csr']:.2f}    "
            f"{ret}"
        )

    lines.append(sep)
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Step 1: load data ─────────────────────────────────────────────────────
    seeds = load_rocstories_subset(CSV_PATH)
    seeds = seeds[:N]

    if not seeds:
        print("[ERROR] No seeds loaded.", file=sys.stderr)
        sys.exit(1)

    # ── Step 2: initialise storage ────────────────────────────────────────────
    results = {system: [] for system, _ in RUNNERS}

    # ── Step 3: loop over seeds ───────────────────────────────────────────────
    for i, seed_text in enumerate(seeds):
        # Detect start emotion and plan trajectory
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
        except Exception:
            trajectory = [start_emotion] * K

        # Run all four systems
        for system_name, runner in RUNNERS:
            try:
                metrics, _, _, _ = runner(seed_text, trajectory)
            except Exception:
                metrics = dict(ZERO_METRICS)

            results[system_name].append(metrics)

    # ── Step 4: aggregate ─────────────────────────────────────────────────────
    aggregated = {system: _aggregate(entries) for system, entries in results.items()}

    # ── Step 5: print final table ─────────────────────────────────────────────
    table = _format_table(aggregated)
    print(table)

    # ── Step 7: save to file ──────────────────────────────────────────────────
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as fh:
        fh.write(table + "\n")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        close_driver()