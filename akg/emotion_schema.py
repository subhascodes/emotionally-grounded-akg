"""
Affective Knowledge Graph (AKG) — Emotion Schema
==================================================

This module defines a psychologically defensible subset of OCC (Ortony, Clore &
Collins, 1988) emotions selected for constrained emotional progression modeling
in short narrative contexts (2–3 sentence story continuations).

Selection rationale
-------------------
The OCC model organises emotions around three appraisal foci: *events*
(consequences for goals), *agents* (actions/accountability), and *objects*
(properties/standards). From this taxonomy we retain only emotions that:

1. **Arise quickly** — they can be credibly established within one or two
   narrative sentences without requiring extended backstory (ruling out
   long-term dispositional states such as *love*, *hate*, or *satisfaction*).

2. **Are perceptually discriminable in short text** — each emotion maps onto a
   distinct surface-level linguistic and situational signature, reducing
   annotation ambiguity and supporting measurable inter-rater agreement.

3. **Cover the core valence × appraisal-focus space** — the nine emotions
   span positive/negative valence and the three OCC foci (event-based,
   agent-based, object-based), giving the graph enough structural diversity to
   model meaningful progression arcs without combinatorial explosion.

4. **Support tractable transition constraints** — the set is small enough that
   each node can be assigned at most 3–4 psychologically plausible outgoing
   edges, keeping the eventual transition graph sparse and interpretable.

Excluded categories
-------------------
* *Fortunes-of-others* emotions (e.g., happy-for, resentment, gloating, pity)
  were excluded because they require modelling a second agent's goal structure,
  which is rarely recoverable from 2–3 sentence contexts.
* Fine-grained variants within the same appraisal branch (e.g., distress vs.
  fear, or pride vs. admiration) were collapsed to the more narratively
  frequent representative to avoid near-duplicate nodes.
* *Satisfaction* and *relief* were excluded as they presuppose completed event
  sequences that exceed the short-context window assumption.

Reference
---------
Ortony, A., Clore, G. L., & Collins, A. (1988). *The Cognitive Structure of
Emotions*. Cambridge University Press.
"""

# ---------------------------------------------------------------------------
# Core emotion list
# ---------------------------------------------------------------------------

EMOTION_LIST: list[str] = [
    "joy",
    "distress",
    "hope",
    "fear",
    "pride",
    "shame",
    "anger",
    "gratitude",
    "surprise",
]

# ---------------------------------------------------------------------------
# Academic descriptions grounded in OCC appraisal theory
# ---------------------------------------------------------------------------

EMOTION_DESCRIPTIONS: dict[str, str] = {
    "joy": (
        "An event-based positive emotion arising from the appraisal that a "
        "desirable event has occurred and is consistent with the agent's active "
        "goals (OCC: pleased about a goal-relevant event). Intensity scales with "
        "goal importance and event desirability."
    ),
    "distress": (
        "An event-based negative emotion arising from the appraisal that an "
        "undesirable event has occurred and is inconsistent with the agent's "
        "active goals (OCC: displeased about a goal-relevant event). Serves as "
        "the valence complement of joy within the well-being branch."
    ),
    "hope": (
        "An event-based positive prospective emotion arising from the appraisal "
        "that a desirable event is possible but not yet confirmed (OCC: pleased "
        "about a prospective desirable event). Distinguished from joy by its "
        "future orientation and inherent uncertainty."
    ),
    "fear": (
        "An event-based negative prospective emotion arising from the appraisal "
        "that an undesirable event is possible but not yet confirmed (OCC: "
        "displeased about a prospective undesirable event). Serves as the "
        "valence complement of hope within the prospect branch."
    ),
    "pride": (
        "An agent-based positive self-directed emotion arising from the appraisal "
        "that one's own praiseworthy action is consistent with personal or social "
        "standards (OCC: approving of one's own action). Requires attribution of "
        "the action to the self."
    ),
    "shame": (
        "An agent-based negative self-directed emotion arising from the appraisal "
        "that one's own blameworthy action violates personal or social standards "
        "(OCC: disapproving of one's own action). Distinguished from guilt by its "
        "public or social evaluative dimension in most narrative instantiations."
    ),
    "anger": (
        "An agent-based negative other-directed emotion arising from the appraisal "
        "that another agent has performed a blameworthy action that is harmful or "
        "unjust to the self or a valued party (OCC: displeased about an event AND "
        "disapproving of the causal agent). Combines event-based and agent-based "
        "appraisal branches."
    ),
    "gratitude": (
        "An agent-based positive other-directed emotion arising from the appraisal "
        "that another agent has performed a praiseworthy action that is beneficial "
        "to the self (OCC: pleased about an event AND approving of the causal "
        "agent). Serves as the valence complement of anger in the agent branch."
    ),
    "surprise": (
        "A valence-neutral event-based emotion arising from the appraisal that an "
        "event is highly unexpected relative to the agent's world model, "
        "irrespective of whether the event is desirable or undesirable. Included "
        "as a transitional emotion that modulates the direction of subsequent "
        "appraisal and is frequently evidenced in short narrative pivot points."
    ),
}

# ---------------------------------------------------------------------------
# Sanity check (importable guard)
# ---------------------------------------------------------------------------

assert set(EMOTION_LIST) == set(EMOTION_DESCRIPTIONS.keys()), (
    "EMOTION_LIST and EMOTION_DESCRIPTIONS are out of sync. "
    "Every emotion must have exactly one description entry."
)