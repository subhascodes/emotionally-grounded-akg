"""
scripts/create_emotion_nodes.py
================================

Constructs or updates ``Emotion`` nodes in the Neo4j AKG graph.

Each node corresponds to one of the nine OCC emotions defined in
``akg/emotion_schema.py`` and stores the following properties:

* ``name`` (str) — canonical emotion identifier; serves as the unique key.
* ``description`` (str) — academic OCC-grounded description from
  ``EMOTION_DESCRIPTIONS``.

Idempotency
-----------
Node creation uses Cypher ``MERGE`` on the ``name`` property, followed by
``SET`` to overwrite ``description``.  Running this script multiple times
against the same Neo4j instance is therefore safe: existing nodes are updated
in place rather than duplicated.

A uniqueness constraint on ``Emotion.name`` is created before any node is
written.  In Neo4j 5.x ``CREATE CONSTRAINT IF NOT EXISTS`` is idempotent and
will not error if the constraint already exists.

Dependencies
------------
* ``neo4j`` (official Python driver, ≥ 5.0)
* ``python-dotenv``
* ``akg/emotion_schema.py`` must be importable from the project root.
* ``scripts/neo4j_connection.py`` must be importable from the same root.

Usage
-----
Run from the project root::

    python -m scripts.create_emotion_nodes

or::

    python scripts/create_emotion_nodes.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path when script is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neo4j import Driver, Session

from akg.emotion_schema import EMOTION_LIST, EMOTION_DESCRIPTIONS
from scripts.neo4j_connection import get_driver, verify_connectivity

# ---------------------------------------------------------------------------
# Cypher templates
# ---------------------------------------------------------------------------

_CREATE_CONSTRAINT_CYPHER: str = """
CREATE CONSTRAINT emotion_name_unique IF NOT EXISTS
FOR (e:Emotion)
REQUIRE e.name IS UNIQUE
"""

_MERGE_NODE_CYPHER: str = """
MERGE (e:Emotion {name: $name})
SET e.description = $description
RETURN e.name AS name
"""


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def ensure_constraint(session: Session) -> None:
    """Create a uniqueness constraint on ``Emotion.name`` if absent.

    Uses ``CREATE CONSTRAINT IF NOT EXISTS`` (Neo4j 5.x syntax) so the
    operation is idempotent.

    Parameters
    ----------
    session:
        An open Neo4j session with write privileges.
    """
    session.run(_CREATE_CONSTRAINT_CYPHER)
    print("[create_emotion_nodes] Uniqueness constraint on Emotion.name ensured.")


def create_emotion_nodes(driver: Driver, database: str = "neo4j") -> None:
    """Merge all OCC emotion nodes into the graph.

    Iterates over ``EMOTION_LIST`` and writes one ``Emotion`` node per entry,
    setting ``name`` and ``description`` from ``EMOTION_DESCRIPTIONS``.
    Uses ``MERGE`` so re-runs are safe.

    Parameters
    ----------
    driver:
        An authenticated, open :class:`neo4j.Driver` instance.
    database:
        Target Neo4j database name (default: ``"neo4j"``).
    """
    with driver.session(database=database) as session:
        ensure_constraint(session)

        created: int = 0
        for emotion in EMOTION_LIST:
            result = session.run(
                _MERGE_NODE_CYPHER,
                name=emotion,
                description=EMOTION_DESCRIPTIONS[emotion],
            )
            record = result.single()
            if record:
                print(f"[create_emotion_nodes]   MERGE Emotion({record['name']})")
                created += 1

        print(
            f"[create_emotion_nodes] Done. "
            f"{created}/{len(EMOTION_LIST)} nodes merged."
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point: verify connectivity then build emotion nodes."""
    verify_connectivity()
    with get_driver() as driver:
        create_emotion_nodes(driver)


if __name__ == "__main__":
    main()