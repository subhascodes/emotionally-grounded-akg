"""
scripts/create_transition_edges.py
====================================

Constructs or updates ``TRANSITIONS_TO`` relationships between ``Emotion``
nodes in the Neo4j AKG graph.

Each directed edge corresponds to one allowed short-term emotional transition
defined in ``akg/transition_matrix.py`` and stores the following properties:

* ``appraisal_condition`` (str) — the OCC-grounded cognitive event that
  licenses the transition.
* ``behavioral_tendency`` (str) — the action-readiness tendency associated
  with arriving at the target emotion via this specific transition.

Idempotency
-----------
Relationship creation uses Cypher ``MATCH`` to locate the two endpoint nodes,
then ``MERGE`` on the relationship itself.  ``SET`` overwrites metadata
properties on each run.  Running this script multiple times is therefore safe:
relationships are updated in place, never duplicated.

Pre-condition
-------------
Both endpoint ``Emotion`` nodes **must already exist** before this script
runs.  Execute ``create_emotion_nodes.py`` first.  If a source or target node
is missing, the ``MATCH`` clause will silently skip that edge; a warning is
printed for each such case.

Dependencies
------------
* ``neo4j`` (official Python driver, ≥ 5.0)
* ``python-dotenv``
* ``akg/transition_matrix.py`` must be importable from the project root.
* ``scripts/neo4j_connection.py`` must be importable from the same root.

Usage
-----
Run from the project root after ``create_emotion_nodes.py``::

    python -m scripts.create_transition_edges

or::

    python scripts/create_transition_edges.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path when script is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neo4j import Driver, Session

from akg.transition_matrix import TRANSITIONS
from scripts.neo4j_connection import get_driver, verify_connectivity

# ---------------------------------------------------------------------------
# Cypher templates
# ---------------------------------------------------------------------------

# MERGE the relationship between two already-existing Emotion nodes.
# The MATCH ensures we never create dangling relationships or phantom nodes.
# SET overwrites metadata so re-runs reflect any upstream changes to
# transition_matrix.py without leaving stale property values.
_MERGE_EDGE_CYPHER: str = """
MATCH (src:Emotion {name: $src_name})
MATCH (tgt:Emotion {name: $tgt_name})
MERGE (src)-[r:TRANSITIONS_TO {src: $src_name, tgt: $tgt_name}]->(tgt)
SET   r.appraisal_condition  = $appraisal_condition,
      r.behavioral_tendency  = $behavioral_tendency
RETURN src.name AS src, tgt.name AS tgt
"""

# Lightweight existence check used for pre-flight warnings.
_CHECK_NODE_CYPHER: str = """
MATCH (e:Emotion {name: $name})
RETURN e.name AS name
"""


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def _node_exists(session: Session, name: str) -> bool:
    """Return True if an ``Emotion`` node with the given *name* exists.

    Parameters
    ----------
    session:
        An open Neo4j session.
    name:
        Emotion name to look up.
    """
    result = session.run(_CHECK_NODE_CYPHER, name=name)
    return result.single() is not None


def create_transition_edges(driver: Driver, database: str = "neo4j") -> None:
    """Merge all OCC transition relationships into the graph.

    Iterates over the nested ``TRANSITIONS`` dictionary and writes one
    ``TRANSITIONS_TO`` relationship per entry.  Each relationship carries
    ``appraisal_condition`` and ``behavioral_tendency`` as properties.

    Missing endpoint nodes are reported as warnings; processing continues
    for all remaining edges.

    Parameters
    ----------
    driver:
        An authenticated, open :class:`neo4j.Driver` instance.
    database:
        Target Neo4j database name (default: ``"neo4j"``).
    """
    total_attempted: int = 0
    total_merged: int = 0
    total_skipped: int = 0

    with driver.session(database=database) as session:

        for src_name, targets in TRANSITIONS.items():

            # Pre-flight: warn if source node is absent.
            if not _node_exists(session, src_name):
                print(
                    f"[create_transition_edges] WARNING: source node "
                    f"Emotion({src_name!r}) not found — "
                    f"run create_emotion_nodes.py first. "
                    f"Skipping {len(targets)} edge(s)."
                )
                total_skipped += len(targets)
                total_attempted += len(targets)
                continue

            for tgt_name, meta in targets.items():
                total_attempted += 1

                # Pre-flight: warn if target node is absent.
                if not _node_exists(session, tgt_name):
                    print(
                        f"[create_transition_edges] WARNING: target node "
                        f"Emotion({tgt_name!r}) not found — skipping edge "
                        f"{src_name!r} -> {tgt_name!r}."
                    )
                    total_skipped += 1
                    continue

                result = session.run(
                    _MERGE_EDGE_CYPHER,
                    src_name=src_name,
                    tgt_name=tgt_name,
                    appraisal_condition=meta["appraisal_condition"],
                    behavioral_tendency=meta["behavioral_tendency"],
                )
                record = result.single()
                if record:
                    print(
                        f"[create_transition_edges]   MERGE "
                        f"({record['src']})-[:TRANSITIONS_TO]->({record['tgt']})"
                    )
                    total_merged += 1

    print(
        f"[create_transition_edges] Done. "
        f"{total_merged} edges merged, "
        f"{total_skipped} skipped, "
        f"{total_attempted} attempted."
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point: verify connectivity then build transition edges."""
    verify_connectivity()
    with get_driver() as driver:
        create_transition_edges(driver)


if __name__ == "__main__":
    main()