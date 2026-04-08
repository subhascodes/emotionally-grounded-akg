"""
scripts/create_transition_edges.py

Creates directed TRANSITION edges between Emotion nodes in the Neo4j AKG.
Each edge is annotated with OCC-grounded appraisal and behavioral tendency
properties.  All transitions are OCC-theory-consistent.

Entry point: main()
"""

from neo4j import Driver
from akg.neo4j_connector import get_driver

# ── OCC-consistent transition data ────────────────────────────────────────────
TRANSITIONS = {
    # ── distress ──────────────────────────────────────────────────────────────
    "distress": {
        "shame": {
            "appraisal": "self-blame after failure",
            "behavior":  "withdrawal, avoidance",
        },
        "anger": {
            "appraisal": "external blame attribution",
            "behavior":  "aggression, confrontation",
        },
        "hope": {
            "appraisal": "belief in improvement",
            "behavior":  "effort, persistence",
        },
        "fear": {
            "appraisal": "anticipation of continued harm",
            "behavior":  "hypervigilance, escape-seeking",
        },
    },

    # ── shame ─────────────────────────────────────────────────────────────────
    "shame": {
        "distress": {
            "appraisal": "re-exposure to the failure event",
            "behavior":  "rumination, self-criticism",
        },
        "anger": {
            "appraisal": "blame redirected outward to deflect self-judgment",
            "behavior":  "hostile defensiveness, lashing out",
        },
        "hope": {
            "appraisal": "recovery of self-worth through corrective action",
            "behavior":  "reparative effort, self-improvement",
        },
        "pride": {
            "appraisal": "recognition of growth beyond past failure",
            "behavior":  "renewed self-confidence, forward engagement",
        },
    },

    # ── anger ─────────────────────────────────────────────────────────────────
    "anger": {
        "distress": {
            "appraisal": "anger exhausts into sadness as agency is lost",
            "behavior":  "withdrawal, resignation",
        },
        "shame": {
            "appraisal": "reflection reveals one's own role in the conflict",
            "behavior":  "self-reproach, social retreat",
        },
        "pride": {
            "appraisal": "successful assertion of boundaries or values",
            "behavior":  "confident self-expression, boundary maintenance",
        },
        "fear": {
            "appraisal": "threat escalates beyond perceived control",
            "behavior":  "defensive avoidance, alarm response",
        },
    },

    # ── hope ──────────────────────────────────────────────────────────────────
    "hope": {
        "joy": {
            "appraisal": "anticipated positive outcome is confirmed",
            "behavior":  "celebration, energized engagement",
        },
        "distress": {
            "appraisal": "expected improvement fails to materialise",
            "behavior":  "disappointment, disengagement",
        },
        "pride": {
            "appraisal": "hopeful effort is validated by personal achievement",
            "behavior":  "self-affirmation, increased ambition",
        },
        "gratitude": {
            "appraisal": "support from others enables hoped-for progress",
            "behavior":  "thankfulness, strengthened social bonds",
        },
    },

    # ── fear ──────────────────────────────────────────────────────────────────
    "fear": {
        "distress": {
            "appraisal": "feared event occurs and causes harm",
            "behavior":  "grief, helplessness",
        },
        "hope": {
            "appraisal": "perceived possibility of avoiding the threat",
            "behavior":  "cautious optimism, preparatory action",
        },
        "anger": {
            "appraisal": "fear transforms into resistance when flight is impossible",
            "behavior":  "defensive aggression, defiance",
        },
        "shame": {
            "appraisal": "fear of social judgment about one's cowardice or exposure",
            "behavior":  "concealment, social withdrawal",
        },
    },

    # ── joy ───────────────────────────────────────────────────────────────────
    "joy": {
        "pride": {
            "appraisal": "positive outcome attributed to one's own ability",
            "behavior":  "self-promotion, confidence display",
        },
        "gratitude": {
            "appraisal": "positive outcome attributed to another agent's help",
            "behavior":  "thanks-giving, prosocial reciprocity",
        },
        "hope": {
            "appraisal": "current positive state motivates anticipation of more",
            "behavior":  "forward planning, continued engagement",
        },
        "distress": {
            "appraisal": "joyful state is threatened or suddenly lost",
            "behavior":  "grief, longing",
        },
    },

    # ── pride ─────────────────────────────────────────────────────────────────
    "pride": {
        "joy": {
            "appraisal": "pride expands into general positive affect",
            "behavior":  "exuberance, social sharing",
        },
        "shame": {
            "appraisal": "high standards make subsequent failure more salient",
            "behavior":  "self-reproach, social concealment",
        },
        "gratitude": {
            "appraisal": "achievement acknowledged as partly enabled by others",
            "behavior":  "humble thanks, collaborative orientation",
        },
        "anger": {
            "appraisal": "pride is threatened or disrespected by another",
            "behavior":  "indignation, assertive retaliation",
        },
    },

    # ── gratitude ─────────────────────────────────────────────────────────────
    "gratitude": {
        "joy": {
            "appraisal": "gratitude deepens into positive relational affect",
            "behavior":  "warmth, increased social engagement",
        },
        "hope": {
            "appraisal": "support received encourages belief in future well-being",
            "behavior":  "optimistic planning, reliance on social network",
        },
        "distress": {
            "appraisal": "benefactor's help is withdrawn or proves insufficient",
            "behavior":  "loss, longing for prior support",
        },
        "pride": {
            "appraisal": "internalisation of gratitude motivates self-improvement",
            "behavior":  "striving to be worthy of help received",
        },
    },
}


# ── Connectivity verification ─────────────────────────────────────────────────

def verify_connectivity() -> None:
    """
    Verify Neo4j connectivity using the shared singleton driver.
    Does NOT close the driver — lifecycle is managed by main().

    Raises:
        RuntimeError: If Neo4j is unreachable or credentials are wrong.
    """
    driver = get_driver()
    driver.verify_connectivity()
    print("Neo4j connectivity verified.")


# ── Edge creation ─────────────────────────────────────────────────────────────

def create_edges(driver: Driver) -> None:
    """
    Remove OCC-inconsistent edges, then MERGE all OCC-consistent TRANSITION
    edges into Neo4j.

    The driver is caller-managed — this function must NOT close it.

    Args:
        driver (Driver): Open Neo4j driver instance.
    """
    # ── Step 1: remove invalid transitions ───────────────────────────────────
    print("Removing invalid OCC transitions...")

    remove_cypher = (
        "MATCH (a:Emotion)-[r:TRANSITION]->(b:Emotion) "
        "WHERE (a.name = 'distress' AND b.name = 'gratitude') "
        "   OR (a.name = 'anger'    AND b.name = 'gratitude') "
        "   OR (a.name = 'fear'     AND b.name = 'gratitude') "
        "DELETE r"
    )

    with driver.session() as session:
        session.run(remove_cypher)

    for src, tgt in [("distress", "gratitude"), ("anger", "gratitude"), ("fear", "gratitude")]:
        print(f"  [REMOVED] {src} → {tgt}")

    # ── Step 2: create / update OCC-consistent transitions ───────────────────
    print("\nCreating OCC-consistent TRANSITION edges...")

    merge_cypher = (
        "MATCH (a:Emotion {name: $src}) "
        "MATCH (b:Emotion {name: $tgt}) "
        "MERGE (a)-[r:TRANSITION]->(b) "
        "SET r.appraisal = $appraisal, "
        "    r.behavior  = $behavior"
    )

    errors = []

    with driver.session() as session:
        for src, targets in TRANSITIONS.items():
            for tgt, meta in targets.items():
                try:
                    session.run(
                        merge_cypher,
                        {
                            "src":       src,
                            "tgt":       tgt,
                            "appraisal": meta["appraisal"],
                            "behavior":  meta["behavior"],
                        },
                    )
                    print(f"  [OK] {src} → {tgt}")
                except Exception as exc:
                    errors.append(f"{src} → {tgt}: {exc}")

    if errors:
        raise RuntimeError(
            f"{len(errors)} edge(s) failed to create:\n" + "\n".join(errors)
        )


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    """
    Verify connectivity then create all AKG transition edges using a single
    driver that remains open for all database operations.
    """
    print("=" * 60)
    print("CREATE TRANSITION EDGES — AKG Setup")
    print("=" * 60)
    print()

    verify_connectivity()
    print()

    driver = get_driver()
    try:
        create_edges(driver)
    finally:
        driver.close()

    print()
    print("=" * 60)
    print("AKG transition edges ready.")
    print("=" * 60)


if __name__ == "__main__":
    main()