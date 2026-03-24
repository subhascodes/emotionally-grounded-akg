"""
scripts/neo4j_connection.py
============================

Provides a reusable, context-managed Neo4j driver factory for the Affective
Knowledge Graph (AKG) construction scripts.

Responsibilities
----------------
* Load ``NEO4J_URI``, ``NEO4J_USER``, and ``NEO4J_PASSWORD`` from a ``.env``
  file located at the project root (one level above ``scripts/``).
* Expose a ``get_driver()`` context manager that yields an authenticated
  ``neo4j.Driver`` instance and guarantees it is closed on exit, even if an
  exception is raised.
* Expose a ``verify_connectivity()`` helper for use in CLI entry points.

Environment variables (``.env``)
---------------------------------
::

    NEO4J_URI=bolt://localhost:7687
    NEO4J_USER=neo4j
    NEO4J_PASSWORD=your_password_here

Dependencies
------------
* ``neo4j`` (official Python driver, ≥ 5.0)
* ``python-dotenv``

Notes
-----
This module contains no graph-construction logic.  It is imported by
``create_emotion_nodes.py`` and ``create_transition_edges.py`` as a shared
connectivity layer.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from dotenv import load_dotenv
from neo4j import GraphDatabase, Driver

# ---------------------------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------------------------

# Resolve .env relative to the project root (parent of the scripts/ directory).
_ENV_PATH: Path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH)

_NEO4J_URI: str = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
_NEO4J_USER: str = os.environ.get("NEO4J_USER", "neo4j")
_NEO4J_PASSWORD: str = os.environ.get("NEO4J_PASSWORD", "")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@contextmanager
def get_driver() -> Generator[Driver, None, None]:
    """Context manager that yields an authenticated Neo4j :class:`Driver`.

    Reads connection parameters from environment variables populated by
    ``.env``.  The driver is closed automatically when the ``with`` block
    exits, regardless of whether an exception occurred.

    Yields
    ------
    neo4j.Driver
        An open, authenticated driver connected to the Neo4j instance
        specified by ``NEO4J_URI``.

    Raises
    ------
    neo4j.exceptions.ServiceUnavailable
        If the Neo4j instance cannot be reached at the configured URI.
    neo4j.exceptions.AuthError
        If the supplied credentials are rejected by the server.

    Example
    -------
    ::

        from scripts.neo4j_connection import get_driver

        with get_driver() as driver:
            with driver.session(database="neo4j") as session:
                result = session.run("RETURN 1 AS n")
                print(result.single()["n"])  # 1
    """
    driver: Driver = GraphDatabase.driver(
        _NEO4J_URI,
        auth=(_NEO4J_USER, _NEO4J_PASSWORD),
    )
    try:
        yield driver
    finally:
        driver.close()


def verify_connectivity() -> None:
    """Open a driver, verify server connectivity, then close it.

    Intended for use as a pre-flight check at the top of construction
    scripts.  Prints a confirmation message on success and raises on failure.

    Raises
    ------
    neo4j.exceptions.ServiceUnavailable
        If the server is not reachable.
    neo4j.exceptions.AuthError
        If authentication fails.
    """
    with get_driver() as driver:
        driver.verify_connectivity()
    print(f"[neo4j_connection] Connected successfully to {_NEO4J_URI}")