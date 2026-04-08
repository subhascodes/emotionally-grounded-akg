"""
akg/neo4j_connector.py

Neo4j interface layer for the Affective Knowledge Graph (AKG).
Provides a singleton driver, connectivity verification, neighbor lookup,
transition metadata retrieval, and edge validation.

Configuration is loaded from a .env file via python-dotenv.
"""

import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

# ── Load environment variables ────────────────────────────────────────────────
load_dotenv()

NEO4J_URI      = os.environ.get("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER     = os.environ.get("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "password")

# ── Singleton driver ──────────────────────────────────────────────────────────
_driver = None


def get_driver():
    """
    Return the singleton Neo4j driver, creating it on the first call.

    Returns:
        neo4j.Driver: Active driver instance.

    Raises:
        RuntimeError: If the driver cannot be initialised.
    """
    global _driver
    if _driver is None:
        try:
            _driver = GraphDatabase.driver(
                NEO4J_URI,
                auth=(NEO4J_USER, NEO4J_PASSWORD),
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to create Neo4j driver: {exc}"
            ) from exc
    return _driver


def close_driver() -> None:
    """
    Close the singleton driver and release all associated resources.

    Safe to call even if the driver was never opened.
    """
    global _driver
    if _driver is not None:
        _driver.close()
        _driver = None


# ── Connectivity verification ─────────────────────────────────────────────────

def verify_connectivity() -> None:
    """
    Verify that the Neo4j instance is reachable and accepting queries.

    Raises:
        RuntimeError: If the connectivity check fails for any reason.
    """
    try:
        with get_driver().session() as session:
            session.run("RETURN 1")
    except Exception as exc:
        raise RuntimeError(
            f"Neo4j connectivity check failed: {exc}"
        ) from exc


# ── Internal query helper ─────────────────────────────────────────────────────

def _run_query(cypher: str, parameters: dict = None) -> list:
    """
    Execute a read query and return all result records.

    Args:
        cypher     (str):  Cypher query string.
        parameters (dict): Query parameters (default: empty dict).

    Returns:
        list[neo4j.Record]: All records returned by the query.

    Raises:
        RuntimeError: If the driver raises any exception during execution.
    """
    parameters = parameters or {}
    try:
        with get_driver().session() as session:
            result = session.run(cypher, parameters)
            return list(result)
    except Exception as exc:
        raise RuntimeError(f"Neo4j query failed: {exc}") from exc


# ── Public interface ──────────────────────────────────────────────────────────

def get_neighbors(emotion: str) -> list:
    """
    Return all emotions reachable from *emotion* via a TRANSITION edge.

    Cypher:
        MATCH (a:Emotion {name: $emotion})-[:TRANSITION]->(b:Emotion)
        RETURN b.name AS neighbor

    Args:
        emotion (str): Source emotion node name.

    Returns:
        list[str]: Sorted list of neighbour emotion names.
                   Returns an empty list when the node has no outgoing edges
                   or does not exist in the graph.
    """
    cypher = (
        "MATCH (a:Emotion {name: $emotion})-[:TRANSITION]->(b:Emotion) "
        "RETURN b.name AS neighbor"
    )
    try:
        records = _run_query(cypher, {"emotion": emotion})
        return sorted(record["neighbor"] for record in records)
    except Exception:
        return []


def get_transition(src: str, tgt: str) -> dict:
    """
    Retrieve metadata for the TRANSITION edge between *src* and *tgt*.

    Cypher:
        MATCH (a:Emotion {name: $src})-[r:TRANSITION]->(b:Emotion {name: $tgt})
        RETURN r.appraisal AS appraisal, r.behavior AS behavior

    Args:
        src (str): Source emotion node name.
        tgt (str): Target emotion node name.

    Returns:
        dict: {"appraisal": str, "behavior": str} when the edge exists.
              Returns an empty dict when no such edge is found.
    """
    cypher = (
        "MATCH (a:Emotion {name: $src})-[r:TRANSITION]->(b:Emotion {name: $tgt}) "
        "RETURN r.appraisal AS appraisal, r.behavior AS behavior"
    )
    try:
        records = _run_query(cypher, {"src": src, "tgt": tgt})
        if not records:
            return {}
        record = records[0]
        return {
            "appraisal": record["appraisal"],
            "behavior":  record["behavior"],
        }
    except Exception:
        return {}


def edge_exists(src: str, dst: str) -> bool:
    """
    Return True when a direct TRANSITION edge exists from *src* to *dst*.

    Cypher:
        MATCH (a:Emotion {name: $src})-[r:TRANSITION]->(b:Emotion {name: $dst})
        RETURN COUNT(r) > 0 AS exists

    Args:
        src (str): Source emotion node name.
        dst (str): Target emotion node name.

    Returns:
        bool: True if the edge exists, False otherwise or on any failure.
    """
    cypher = (
        "MATCH (a:Emotion {name: $src})-[r:TRANSITION]->(b:Emotion {name: $dst}) "
        "RETURN COUNT(r) > 0 AS exists"
    )
    try:
        records = _run_query(cypher, {"src": src, "dst": dst})
        if not records:
            return False
        return bool(records[0]["exists"])
    except Exception:
        return False


def is_valid_transition(src: str, tgt: str) -> bool:
    """
    Alias for edge_exists. Returns True when a TRANSITION edge exists.

    Args:
        src (str): Source emotion node name.
        tgt (str): Target emotion node name.

    Returns:
        bool: True if the edge exists, False otherwise.
    """
    return edge_exists(src, tgt)