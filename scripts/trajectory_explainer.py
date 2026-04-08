"""
scripts/trajectory_explainer.py
================================
Traceability and explanation module for emotional trajectories.

Queries the Neo4j AKG for appraisal and behavioral metadata on each
consecutive transition in a trajectory and generates a Cypher query
that exactly reproduces that trajectory as a visual path in Neo4j.
"""

from akg.neo4j_connector import get_transition


def _build_cypher_query(trajectory: list) -> str:
    """
    Generate a single-line Cypher MATCH query that visualises the exact
    trajectory as a directed path through the AKG.

    Variable names are assigned alphabetically: a, b, c, d, ...
    Works for any trajectory length.

    Example for ["distress", "shame", "hope"]:
        MATCH (a:Emotion {name: "distress"})-[:TRANSITION]->
              (b:Emotion {name: "shame"})-[:TRANSITION]->
              (c:Emotion {name: "hope"})
        RETURN a,b,c

    Args:
        trajectory (list[str]): Ordered list of OCC emotion labels.

    Returns:
        str: Single-line Cypher query string.
    """
    # Generate variable names: a, b, c, ..., z, aa, ab, ...
    def _var(idx: int) -> str:
        letters = "abcdefghijklmnopqrstuvwxyz"
        if idx < 26:
            return letters[idx]
        return letters[idx // 26 - 1] + letters[idx % 26]

    node_parts = []
    for idx, emotion in enumerate(trajectory):
        var   = _var(idx)
        node  = f'({var}:Emotion {{name: "{emotion}"}})'
        node_parts.append(node)

    match_path = "-[:TRANSITION]->".join(node_parts)
    return_vars = ",".join(_var(idx) for idx in range(len(trajectory)))

    return f"MATCH {match_path} RETURN {return_vars}, r"


def explain_trajectory(trajectory: list) -> tuple:
    """
    Produce a step-by-step explanation for each transition in *trajectory*,
    together with a Cypher query that visualises the path in Neo4j.

    For each consecutive pair (prev, next) the function queries the Neo4j AKG
    via the singleton driver (no repeated open/close) and returns the appraisal
    condition and behavioral tendency stored on the TRANSITION edge.

    Args:
        trajectory (list[str]): Ordered list of OCC emotion labels.
                                Must contain at least 2 elements.

    Returns:
        tuple:
            explanation  (list[dict]) — one dict per step with keys:
                "step"      – 1-based step index (int)
                "from"      – source emotion label (str)
                "to"        – target emotion label (str)
                "appraisal" – OCC appraisal condition from the AKG (str)
                "behavior"  – OCC behavioral tendency from the AKG (str)
            cypher_query (str) — single-line MATCH / RETURN query that
                exactly traces the trajectory in Neo4j Browser.

    Raises:
        ValueError: If trajectory contains fewer than 2 elements.
    """
    if len(trajectory) < 2:
        raise ValueError(
            "trajectory must contain at least 2 emotions to explain transitions."
        )

    explanation = []

    for step_idx, (prev, nxt) in enumerate(
        zip(trajectory[:-1], trajectory[1:]), start=1
    ):
        meta = get_transition(prev, nxt)
        explanation.append(
            {
                "step":      step_idx,
                "from":      prev,
                "to":        nxt,
                "appraisal": meta.get(
                    "appraisal",
                    f"No appraisal data found for {prev} → {nxt}.",
                ),
                "behavior":  meta.get(
                    "behavior",
                    f"No behavior data found for {prev} → {nxt}.",
                ),
            }
        )

    cypher_query = _build_cypher_query(trajectory)

    return explanation, cypher_query


def print_explanation(explanation: list, cypher_query: str) -> None:
    """
    Print a formatted traceability block followed by the Neo4j
    visualisation query.

    Args:
        explanation  (list[dict]): Output of explain_trajectory()[0].
        cypher_query (str):        Output of explain_trajectory()[1].
    """
    print("------------------------------------------------------------")
    print("TRACEABILITY")
    print("------------------------------------------------------------")
    for entry in explanation:
        print(f"STEP {entry['step']}:")
        print(f"  {entry['from']} → {entry['to']}")
        print(f"  Appraisal : {entry['appraisal']}")
        print(f"  Behavior  : {entry['behavior']}")
        print()

    print("------------------------------------------------------------")
    print("NEO4J VISUALIZATION QUERY")
    print("------------------------------------------------------------")
    print(cypher_query)
    print()