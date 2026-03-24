"""
Affective Knowledge Graph (AKG) — Transition Matrix
=====================================================

This module defines a sparse, psychologically defensible set of short-term
emotional transitions between the nine OCC emotions defined in
``akg/emotion_schema.py``.

Design principles
-----------------
**Psychological immediacy constraint**
    Only transitions that can plausibly occur within a 2–3 sentence narrative
    window are included. Slow-burn or dispositional shifts (e.g., joy → love,
    distress → depression) are excluded.

**Appraisal re-evaluation as transition mechanism**
    Each edge represents a discrete re-appraisal event: new information, an
    agent's action, or a change in event probability triggers a shift in the
    dominant appraisal dimension, producing a new emotion. Transitions are
    therefore *caused*, not merely sequential.

**Sparsity rationale**
    The maximum out-degree of 3 per node enforces that the graph models
    *typical* narrative trajectories rather than exhaustive logical possibility.
    Edges absent from this matrix should be treated as constrained-out in
    downstream validation, not merely as unmodelled.

**Asymmetry policy**
    Bidirectional edges are included only when re-appraisal in both directions
    is independently motivated (e.g., hope <-> fear reflect symmetric
    prospective re-appraisals under new evidence). Mere logical reversibility
    is not sufficient justification.

**Metadata schema**
    Each edge carries two metadata fields:

    * ``appraisal_condition`` -- the OCC-grounded cognitive event that licenses
      the transition (what the agent must appraise for the shift to occur).
    * ``behavioral_tendency`` -- the action-readiness or expressive tendency
      associated with the *arriving* emotion in this specific transition
      context, following Frijda (1986) action-tendency theory.

Total edges: 24

References
----------
Ortony, A., Clore, G. L., & Collins, A. (1988). *The Cognitive Structure of
Emotions*. Cambridge University Press.
Frijda, N. H. (1986). *The Emotions*. Cambridge University Press.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Type aliases for readability
# ---------------------------------------------------------------------------

TransitionMeta = dict[str, str]
EmotionTransitions = dict[str, TransitionMeta]
TransitionMatrix = dict[str, EmotionTransitions]

# ---------------------------------------------------------------------------
# Transition matrix
# ---------------------------------------------------------------------------

TRANSITIONS: TransitionMatrix = {

    # ------------------------------------------------------------------
    # JOY  (out-degree: 3)
    # A positive event-consequential state; destabilised by threat,
    # loss, or the re-attribution of a benefit to another agent.
    # ------------------------------------------------------------------
    "joy": {
        "gratitude": {
            "appraisal_condition": (
                "Agent retrospectively attributes the desirable event to a "
                "praiseworthy action of another agent"
            ),
            "behavioral_tendency": (
                "Approach and affiliative expression toward the benefactor; "
                "verbal acknowledgment of the other's contribution"
            ),
        },
        "pride": {
            "appraisal_condition": (
                "Agent attributes the desirable event to their own praiseworthy "
                "action or competence"
            ),
            "behavioral_tendency": (
                "Self-display and confident posturing; increased willingness "
                "to take on further challenges"
            ),
        },
        "fear": {
            "appraisal_condition": (
                "New information reveals that the positive event is fragile or "
                "that an undesirable counter-event is now possible"
            ),
            "behavioral_tendency": (
                "Protective vigilance; reduced exploratory behaviour to "
                "preserve the valued state"
            ),
        },
    },

    # ------------------------------------------------------------------
    # DISTRESS  (out-degree: 3)
    # A negative event-consequential state; may escalate, reframe, or
    # redirect depending on causal attribution.
    # ------------------------------------------------------------------
    "distress": {
        "anger": {
            "appraisal_condition": (
                "Agent attributes the undesirable event to a blameworthy "
                "intentional action of another agent"
            ),
            "behavioral_tendency": (
                "Confrontational readiness; verbal or physical challenge "
                "directed at the identified culpable agent"
            ),
        },
        "shame": {
            "appraisal_condition": (
                "Agent re-attributes the undesirable event to their own "
                "inadequate or blameworthy action, with perceived social visibility"
            ),
            "behavioral_tendency": (
                "Social withdrawal and concealment; reduced eye contact "
                "and self-silencing"
            ),
        },
        "hope": {
            "appraisal_condition": (
                "Agent appraises a prospective remedial event as possible, "
                "shifting attention from the current loss to a potential recovery"
            ),
            "behavioral_tendency": (
                "Increased goal-directed planning; tentative approach toward "
                "the envisaged remedial outcome"
            ),
        },
    },

    # ------------------------------------------------------------------
    # HOPE  (out-degree: 3)
    # A positive prospective state contingent on uncertain future events;
    # resolves as probability information increases.
    # ------------------------------------------------------------------
    "hope": {
        "joy": {
            "appraisal_condition": (
                "The prospective desirable event is confirmed as having occurred "
                "or as highly certain"
            ),
            "behavioral_tendency": (
                "Release of preparatory tension; celebratory expression and "
                "approach toward the now-realised outcome"
            ),
        },
        "fear": {
            "appraisal_condition": (
                "New evidence substantially reduces the probability of the "
                "desired outcome or raises the likelihood of an opposing threat"
            ),
            "behavioral_tendency": (
                "Inhibition of approach behaviour; heightened monitoring of "
                "threat-relevant environmental cues"
            ),
        },
        "distress": {
            "appraisal_condition": (
                "The prospective desirable event is disconfirmed; the anticipated "
                "goal-congruent outcome fails to materialise"
            ),
            "behavioral_tendency": (
                "Goal disengagement and dejection; reduced motivation to "
                "pursue the now-foreclosed objective"
            ),
        },
    },

    # ------------------------------------------------------------------
    # FEAR  (out-degree: 3)
    # A negative prospective state; resolves via threat removal,
    # confirmation, or active reappraisal of agency.
    # ------------------------------------------------------------------
    "fear": {
        "distress": {
            "appraisal_condition": (
                "The feared undesirable event is confirmed as having occurred, "
                "converting prospective dread into realised loss"
            ),
            "behavioral_tendency": (
                "Passive coping and dejection; withdrawal from goal-pursuit "
                "in the affected domain"
            ),
        },
        "hope": {
            "appraisal_condition": (
                "New information significantly reduces threat probability or "
                "reveals a viable protective response"
            ),
            "behavioral_tendency": (
                "Relaxation of defensive posture; cautious reorientation "
                "toward positive goal pursuit"
            ),
        },
        "anger": {
            "appraisal_condition": (
                "Agent identifies a specific other agent as responsible for "
                "creating or sustaining the threatening situation"
            ),
            "behavioral_tendency": (
                "Shift from flight-oriented to fight-oriented readiness; "
                "confrontational attribution directed outward"
            ),
        },
    },

    # ------------------------------------------------------------------
    # PRIDE  (out-degree: 2)
    # A positive self-directed agent-based state; relatively stable
    # but vulnerable to social feedback or counterfactual re-appraisal.
    # ------------------------------------------------------------------
    "pride": {
        "joy": {
            "appraisal_condition": (
                "The praised action produces a tangible desirable outcome, "
                "grounding self-appraisal in confirmed event-based benefit"
            ),
            "behavioral_tendency": (
                "Continued engagement with the successful domain; "
                "sharing of the positive outcome with relevant others"
            ),
        },
        "shame": {
            "appraisal_condition": (
                "Social audience rejects or ridicules the agent's self-appraisal, "
                "or the agent discovers a disqualifying flaw in their action"
            ),
            "behavioral_tendency": (
                "Rapid self-concealment; reappraisal of the action as "
                "blameworthy rather than praiseworthy"
            ),
        },
    },

    # ------------------------------------------------------------------
    # SHAME  (out-degree: 3)
    # A negative self-directed agent-based state; motivates concealment
    # but may convert to anger or fuel reparative effort.
    # ------------------------------------------------------------------
    "shame": {
        "anger": {
            "appraisal_condition": (
                "Agent externalises blame, reattributing the evaluative threat "
                "to an unfair social standard or a specific accusing other"
            ),
            "behavioral_tendency": (
                "Defensive hostility directed at the perceived source of "
                "the shame-inducing evaluation; justification of own conduct"
            ),
        },
        "distress": {
            "appraisal_condition": (
                "Agent fully accepts the negative self-evaluation without "
                "identifying a reparative path, intensifying the sense of loss"
            ),
            "behavioral_tendency": (
                "Rumination and passive self-criticism; social withdrawal "
                "without active coping"
            ),
        },
        "pride": {
            "appraisal_condition": (
                "Agent successfully executes a reparative action that is "
                "appraised as praiseworthy, overwriting the prior blameworthy act"
            ),
            "behavioral_tendency": (
                "Rehabilitative self-display; motivated re-engagement with "
                "the domain in which the failure occurred"
            ),
        },
    },

    # ------------------------------------------------------------------
    # ANGER  (out-degree: 3)
    # A negative other-directed agent-based state; resolves via
    # accountability shifts, appeasement, or escalation.
    # ------------------------------------------------------------------
    "anger": {
        "distress": {
            "appraisal_condition": (
                "Confrontational action fails or is blocked; agent reappraises "
                "the situation as beyond their control, shifting focus to the loss"
            ),
            "behavioral_tendency": (
                "Resignation and helplessness; reduced motivation to challenge "
                "the offending agent further"
            ),
        },
        "gratitude": {
            "appraisal_condition": (
                "The blamed agent provides an exculpatory explanation or "
                "performs an unsolicited reparative action appraised as praiseworthy"
            ),
            "behavioral_tendency": (
                "De-escalation of confrontational readiness; approach and "
                "affiliative reorientation toward the former target"
            ),
        },
        "shame": {
            "appraisal_condition": (
                "Agent receives credible social or evidentiary feedback that "
                "their anger was unjustified, redirecting blame to the self"
            ),
            "behavioral_tendency": (
                "Withdrawal of accusation; self-critical rumination about "
                "the misjudged attribution"
            ),
        },
    },

    # ------------------------------------------------------------------
    # GRATITUDE  (out-degree: 2)
    # A positive other-directed agent-based state; relatively stable
    # but may sour if the beneficial action is reappraised.
    # ------------------------------------------------------------------
    "gratitude": {
        "joy": {
            "appraisal_condition": (
                "The benefit provided by the other agent produces a confirmed "
                "desirable outcome, anchoring gratitude in event-level gain"
            ),
            "behavioral_tendency": (
                "Reciprocal prosocial behaviour; overt expression of "
                "appreciation and strengthened affiliative bond"
            ),
        },
        "anger": {
            "appraisal_condition": (
                "Agent discovers that the seemingly praiseworthy action was "
                "self-interested, manipulative, or accompanied by a hidden cost"
            ),
            "behavioral_tendency": (
                "Retraction of positive evaluation; confrontational reorientation "
                "toward the now-blamed agent"
            ),
        },
    },

    # ------------------------------------------------------------------
    # SURPRISE  (out-degree: 3)
    # A valence-neutral event-based state triggered by violated
    # expectation; functions as a transitional node that routes
    # subsequent appraisal along positive or negative branches.
    # ------------------------------------------------------------------
    "surprise": {
        "joy": {
            "appraisal_condition": (
                "Unexpected event is rapidly appraised as desirable and "
                "goal-congruent upon secondary evaluation"
            ),
            "behavioral_tendency": (
                "Orienting response followed by approach; spontaneous "
                "expressive vocalisation and increased engagement"
            ),
        },
        "fear": {
            "appraisal_condition": (
                "Unexpected event is rapidly appraised as potentially threatening "
                "or goal-incongruent upon secondary evaluation"
            ),
            "behavioral_tendency": (
                "Startle-to-freeze transition; heightened environmental scanning "
                "and defensive postural adjustment"
            ),
        },
        "distress": {
            "appraisal_condition": (
                "Unexpected event is appraised as an undesirable outcome that "
                "has already negatively affected a valued goal"
            ),
            "behavioral_tendency": (
                "Abrupt cessation of current goal-pursuit; "
                "cognitive reorientation toward loss appraisal"
            ),
        },
    },
}

# ---------------------------------------------------------------------------
# Module-level integrity checks
# ---------------------------------------------------------------------------

_VALID_EMOTIONS: frozenset[str] = frozenset([
    "joy", "distress", "hope", "fear",
    "pride", "shame", "anger", "gratitude", "surprise",
])

for _src, _targets in TRANSITIONS.items():
    assert _src in _VALID_EMOTIONS, f"Unknown source emotion: {_src!r}"
    assert len(_targets) <= 3, (
        f"Out-degree constraint violated for {_src!r}: {len(_targets)} > 3"
    )
    for _tgt, _meta in _targets.items():
        assert _tgt in _VALID_EMOTIONS, (
            f"Unknown target emotion {_tgt!r} in transitions for {_src!r}"
        )
        assert _src != _tgt, f"Self-loop detected at {_src!r}"
        assert "appraisal_condition" in _meta and "behavioral_tendency" in _meta, (
            f"Incomplete metadata on edge {_src!r} -> {_tgt!r}"
        )

_TOTAL_EDGES: int = sum(len(v) for v in TRANSITIONS.values())
assert 20 <= _TOTAL_EDGES <= 25, (
    f"Edge count {_TOTAL_EDGES} outside target range [20, 25]"
)