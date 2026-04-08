"""
scripts/explanation_engine.py
==============================

Graph-grounded explanation layer for the Affective Knowledge Graph (AKG).

This module retrieves ``TRANSITIONS_TO`` relationship metadata stored in Neo4j
and produces structured, deterministic natural-language explanations of
emotional transitions.  No inference, heuristics, or language model calls are
involved: all explanatory content is derived exclusively from properties
written to the graph by ``scripts/create_transition_edges.py``.

Theoretical grounding
---------------------
Each ``TRANSITIONS_TO`` relationship in the AKG stores two properties that
were derived from OCC appraisal theory (Ortony, Clore & Collins, 1988) and
Frijda's (1986) action-tendency framework:

* ``appraisal_condition`` — the cognitive re-evaluation event that licenses
  the transition (what the agent must appraise for the shift to occur).
* ``behavioral_tendency`` — the action-readiness or expressive tendency
  characteristic of arriving at the target emotion via this specific path.

The explanation format is therefore *path-sensitive*: the same target emotion
reached via different source emotions may carry different behavioral tendencies,
accurately reflecting the OCC principle that emotional intensity and expression
depend on the full appraisal context, not merely on the terminal state.

Explanation format
------------------
Successful transitions produce a two-sentence structured explanation::

    "The character transitioned from <src> to <tgt> due to appraisal that
    <appraisal_condition>. This shift is associated with <behavioral_tendency>."

Failed lookups (no edge in the graph) produce a violation message::

    "No valid transition exists from <src> to <tgt> in the AKG constraint
    graph. This pair is not licensed by the OCC-grounded transition matrix."

Module responsibilities
-----------------------
* **Read-only**: this module never writes to or modifies the Neo4j graph.
* **Deterministic**: given identical graph state, identical inputs always
  produce identical outputs.
* **Modular**: ``get_transition_metadata`` and ``generate_explanation`` are
  independently callable; the former is suitable for programmatic use in
  downstream validation pipelines.

Dependencies
------------
* ``neo4j`` (official Python driver, ≥ 5.0)
* ``python-dotenv``
* ``scripts/neo4j_connection.py`` must be importable from the project root.

References
----------
Ortony, A., Clore, G. L., & Collins, A. (1988). *The Cognitive Structure of
Emotions*. Cambridge University Press.
Frijda, N. H. (1986). *The Emotions*. Cambridge University Press.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

# Ensure project root is on sys.path when script is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neo4j import Driver

from scripts.neo4j_connector import get_driver

# ---------------------------------------------------------------------------
# Cypher templates
# ---------------------------------------------------------------------------

_FETCH_EDGE_CYPHER: str = """
MATCH (src:Emotion {name: $src_name})-[r:TRANSITIONS_TO]->(tgt:Emotion {name: $tgt_name})
RETURN r.appraisal_condition  AS appraisal_condition,
       r.behavioral_tendency  AS behavioral_tendency
"""

# ---------------------------------------------------------------------------
# Explanation string templates
# ---------------------------------------------------------------------------

_EXPLANATION_TEMPLATE: str = (
    "The character transitioned from {src} to {tgt} due to appraisal that "
    "{appraisal_condition}. "
    "This shift is associated with {behavioral_tendency}."
)

_VIOLATION_TEMPLATE: str = (
    "No valid transition exists from {src} to {tgt} in the AKG constraint "
    "graph. This pair is not licensed by the OCC-grounded transition matrix."
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_transition_metadata(
    src: str,
    tgt: str,
    driver: Optional[Driver] = None,
    database: str = "neo4j",
) -> Optional[dict[str, str]]:
    """Retrieve stored OCC metadata for a directed emotional transition.

    Executes a read-only Cypher ``MATCH`` against the Neo4j AKG graph to
    fetch the ``appraisal_condition`` and ``behavioral_tendency`` properties
    of the ``TRANSITIONS_TO`` relationship between the two named ``Emotion``
    nodes.

    Parameters
    ----------
    src:
        Name of the source ``Emotion`` node (e.g. ``"distress"``).
    tgt:
        Name of the target ``Emotion`` node (e.g. ``"anger"``).
    driver:
        An already-open :class:`neo4j.Driver` instance.  If ``None``
        (default), the function opens its own driver via
        :func:`scripts.neo4j_connection.get_driver` and closes it before
        returning.  Pass an explicit driver when calling this function in a
        tight loop to avoid per-call connection overhead.
    database:
        Target Neo4j database name (default: ``"neo4j"``).

    Returns
    -------
    dict[str, str] | None
        A dictionary with keys ``"appraisal_condition"`` and
        ``"behavioral_tendency"`` if a directed ``TRANSITIONS_TO`` edge from
        *src* to *tgt* exists in the graph, or ``None`` if no such edge is
        found (including when either node is absent).

    Examples
    --------
    ::

        meta = get_transition_metadata("distress", "anger")
        # {
        #     "appraisal_condition": "Agent attributes the undesirable event ...",
        #     "behavioral_tendency": "Confrontational readiness; ..."
        # }

        meta = get_transition_metadata("joy", "shame")
        # None  — this edge is not present in the AKG transition matrix
    """
    def _run(drv: Driver) -> Optional[dict[str, str]]:
        with drv.session(database=database) as session:
            result = session.run(
                _FETCH_EDGE_CYPHER,
                src_name=src,
                tgt_name=tgt,
            )
            record = result.single()
            if record is None:
                return None
            return {
                "appraisal_condition": record["appraisal_condition"],
                "behavioral_tendency": record["behavioral_tendency"],
            }

    if driver is not None:
        return _run(driver)

    with get_driver() as drv:
        return _run(drv)


def generate_explanation(
    src: str,
    tgt: str,
    driver: Optional[Driver] = None,
    database: str = "neo4j",
) -> str:
    """Generate a structured, deterministic explanation of an emotional transition.

    Retrieves ``TRANSITIONS_TO`` metadata from the Neo4j AKG graph via
    :func:`get_transition_metadata` and formats it into a two-sentence
    academic explanation.  If the transition is not present in the constraint
    graph, a violation message is returned instead.

    The explanation is constructed entirely from stored graph properties; no
    inference, template interpolation beyond property substitution, or
    language model involvement occurs.

    Parameters
    ----------
    src:
        Name of the source ``Emotion`` node (e.g. ``"distress"``).
    tgt:
        Name of the target ``Emotion`` node (e.g. ``"anger"``).
    driver:
        An already-open :class:`neo4j.Driver` instance.  If ``None``
        (default), the function opens its own driver via
        :func:`scripts.neo4j_connection.get_driver`.  Pass an explicit driver
        when batching multiple calls.
    database:
        Target Neo4j database name (default: ``"neo4j"``).

    Returns
    -------
    str
        A deterministic natural-language explanation string.  Two forms:

        *Valid transition*::

            "The character transitioned from distress to anger due to
            appraisal that another agent caused a blameworthy event.
            This shift is associated with confrontational action readiness."

        *Constraint violation*::

            "No valid transition exists from joy to shame in the AKG
            constraint graph. This pair is not licensed by the
            OCC-grounded transition matrix."

    Examples
    --------
    ::

        print(generate_explanation("fear", "anger"))
        # "The character transitioned from fear to anger due to appraisal
        #  that Agent identifies a specific other agent as responsible for
        #  creating or sustaining the threatening situation. This shift is
        #  associated with Shift from flight-oriented to fight-oriented
        #  readiness; confrontational attribution directed outward."

        print(generate_explanation("joy", "shame"))
        # "No valid transition exists from joy to shame in the AKG
        #  constraint graph. This pair is not licensed by the
        #  OCC-grounded transition matrix."
    """
    meta = get_transition_metadata(src=src, tgt=tgt, driver=driver, database=database)

    if meta is None:
        return _VIOLATION_TEMPLATE.format(src=src, tgt=tgt)

    return _EXPLANATION_TEMPLATE.format(
        src=src,
        tgt=tgt,
        appraisal_condition=meta["appraisal_condition"],
        behavioral_tendency=meta["behavioral_tendency"],
    )


# ---------------------------------------------------------------------------
# CLI demo entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Demonstrate the explanation engine against a small set of test pairs.

    Prints explanations for both valid and invalid transitions to verify that
    Neo4j connectivity, metadata retrieval, and string formatting are all
    working correctly end-to-end.
    """
    test_pairs: list[tuple[str, str]] = [
        # Valid transitions from the AKG matrix
        ("distress", "anger"),
        ("hope", "joy"),
        ("fear", "anger"),
        ("shame", "pride"),
        ("anger", "gratitude"),
        ("surprise", "fear"),
        # Invalid / constraint-violating pairs
        ("joy", "shame"),
        ("gratitude", "fear"),
        ("pride", "distress"),
        ("unknown_emotion", "joy"),
    ]

    print("=" * 70)
    print("AKG Explanation Engine — Demo Output")
    print("=" * 70)

    with get_driver() as driver:
        for src, tgt in test_pairs:
            print(f"\n[{src}] --> [{tgt}]")
            print("-" * 50)
            explanation = generate_explanation(src=src, tgt=tgt, driver=driver)
            print(explanation)

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()