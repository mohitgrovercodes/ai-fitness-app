"""
Refusal Threshold — Segment-Targeted Coverage.

Phase 5 of IMPLEMENTATION_PLAN.md.

WHY ABSOLUTE COUNT IS WRONG (§6.1):
  "User has knee + wrist pain → safe_pool = {crunch, russian_twist, plank,
  side_plank, dead_bug, clamshell}. Six items, threshold passes. System builds
  a 'leg day' out of five core exercises and one clamshell. That's not a leg
  day — it's a mockery. The right answer was 'see a physio'."

THE CORRECT STRATEGY (§6.2):
  Each exercise is tagged with a primary anatomical segment. A workout request
  targets one or more segments (derived from the user's goal). A refusal fires
  when any REQUIRED segment has fewer than min_per_segment safe exercises.

  Example: "leg day" targets [LOWER_ANTERIOR, LOWER_POSTERIOR].
    - PFPS + severe wrist injury → safe_pool has 0 LOWER_ANTERIOR exercises
      (all squats/lunges need grip or knee extension).
    - Result: REFUSAL with clear explanation + physio referral.

SEGMENT TAXONOMY:
  7 primary segments cover the full body. Every exercise maps to exactly ONE
  primary segment (the dominant mover); secondary segments are optional tags
  that allow cross-segment queries (e.g., hip thrust is LOWER_POSTERIOR primary,
  may appear in a full-body query via secondary LOWER_ANTERIOR).
"""
from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional, Set

from pydantic import BaseModel, Field

from app.safety.schema import BiomechanicalTags


# ─────────────────────────────────────────────────────────────────────────────
# Segment taxonomy
# ─────────────────────────────────────────────────────────────────────────────

class Segment(str, Enum):
    """
    Seven primary anatomical segments. Each exercise maps to ONE primary
    segment (the dominant muscle group targeted).

    Lower body splits:
      LOWER_ANTERIOR — quads, hip flexors (squats, leg extension, step-up)
      LOWER_POSTERIOR — glutes, hamstrings, calves (RDL, hip thrust, leg curl)
      LOWER_LATERAL   — hip abductors, adductors (clamshell, lateral lunge)

    Core splits:
      CORE_ANTERIOR   — rectus abdominis, obliques (crunch, hanging leg raise)
      CORE_POSTERIOR  — erectors, multifidus (hyperextension, bird-dog)

    Upper body splits:
      UPPER_PUSH — chest, shoulders, triceps (bench, overhead press, push-up)
      UPPER_PULL — back, biceps (row, pull-up, lat pulldown)
    """
    LOWER_ANTERIOR  = "LOWER_ANTERIOR"
    LOWER_POSTERIOR = "LOWER_POSTERIOR"
    LOWER_LATERAL   = "LOWER_LATERAL"
    CORE_ANTERIOR   = "CORE_ANTERIOR"
    CORE_POSTERIOR  = "CORE_POSTERIOR"
    UPPER_PUSH      = "UPPER_PUSH"
    UPPER_PULL      = "UPPER_PULL"


# ─────────────────────────────────────────────────────────────────────────────
# Extend BiomechanicalTags with segment fields
# ─────────────────────────────────────────────────────────────────────────────
# We do NOT inherit from BiomechanicalTags (Pydantic strict mode makes that
# awkward). Instead we define a parallel SegmentedTags model that adds the
# two segment fields. The filter consumes BiomechanicalTags; the refusal layer
# consumes SegmentedTags (which is a superset).

class SegmentedTags(BiomechanicalTags):
    """
    BiomechanicalTags extended with anatomical segment classification.

    primary_segment:    The dominant muscle group this exercise targets.
    secondary_segments: Optional additional segments this exercise meaningfully
                        activates (for cross-training queries like full body).
    """
    primary_segment: Segment
    secondary_segments: List[Segment] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Goal → segment mapping
# ─────────────────────────────────────────────────────────────────────────────

# Standard goal → required segments mapping.
# The workout-planning LLM (or a simple intent classifier) maps the user's
# training goal to one of these keys. The refusal threshold checks all
# required segments have >= min_per_segment safe exercises.

GOAL_SEGMENT_MAP: Dict[str, List[Segment]] = {
    "leg_day": [
        Segment.LOWER_ANTERIOR,
        Segment.LOWER_POSTERIOR,
    ],
    "lower_body": [
        Segment.LOWER_ANTERIOR,
        Segment.LOWER_POSTERIOR,
        Segment.LOWER_LATERAL,
    ],
    "push_day": [
        Segment.UPPER_PUSH,
        Segment.CORE_ANTERIOR,
    ],
    "pull_day": [
        Segment.UPPER_PULL,
        Segment.CORE_POSTERIOR,
    ],
    "upper_body": [
        Segment.UPPER_PUSH,
        Segment.UPPER_PULL,
        Segment.CORE_ANTERIOR,
    ],
    "core": [
        Segment.CORE_ANTERIOR,
        Segment.CORE_POSTERIOR,
    ],
    "full_body": [
        Segment.LOWER_ANTERIOR,
        Segment.LOWER_POSTERIOR,
        Segment.UPPER_PUSH,
        Segment.UPPER_PULL,
        Segment.CORE_ANTERIOR,
    ],
    "active_recovery": [   # minimal — just need at least 1 segment available
        Segment.CORE_ANTERIOR,
    ],
    "rehab": [             # any single segment is acceptable
        Segment.CORE_ANTERIOR,
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# Refusal structures
# ─────────────────────────────────────────────────────────────────────────────

class SegmentCoverage(BaseModel):
    """Per-segment count in the current safe_pool."""
    segment: Segment
    count: int
    is_sufficient: bool


class RefusalDecision(BaseModel):
    """
    The output of maybe_refuse(). If should_refuse is True, the caller must
    surface refusal_message to the user and set WorkoutPlan.refuse=True.
    """
    should_refuse: bool
    missing_segments: List[Segment] = Field(default_factory=list)
    coverage: List[SegmentCoverage] = Field(default_factory=list)
    refusal_message: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# Core refusal algorithm
# ─────────────────────────────────────────────────────────────────────────────

def coverage_by_segment(
    safe_pool: List[SegmentedTags],
) -> Dict[Segment, int]:
    """
    Count how many safe exercises cover each segment.
    Counts primary_segment once + each secondary_segment once.
    """
    counts: Dict[Segment, int] = {seg: 0 for seg in Segment}
    for ex in safe_pool:
        counts[ex.primary_segment] += 1
        for sec in ex.secondary_segments:
            counts[sec] += 1
    return counts


def maybe_refuse(
    safe_pool: List[SegmentedTags],
    goal: str,
    min_per_segment: int = 2,
    custom_target_segments: Optional[List[Segment]] = None,
) -> RefusalDecision:
    """
    Determine whether a safe and balanced workout can be built.

    Args:
        safe_pool: the output of filter_with_audit() (already filtered).
        goal: a key from GOAL_SEGMENT_MAP (e.g. "leg_day", "full_body").
              Ignored if custom_target_segments is provided.
        min_per_segment: minimum exercises required per required segment.
                         Default 2 — prevents single-exercise days.
        custom_target_segments: override the goal map (for dynamic intent).

    Returns:
        RefusalDecision with should_refuse=True + explanation if the safe_pool
        cannot support a balanced workout for the goal, False otherwise.
    """
    # Determine required segments
    if custom_target_segments is not None:
        target_segments = custom_target_segments
    else:
        target_segments = GOAL_SEGMENT_MAP.get(goal, [])

    if not target_segments:
        # Unknown goal — permissive: do not refuse if we don't know the target
        return RefusalDecision(should_refuse=False)

    counts = coverage_by_segment(safe_pool)
    
    # Also count compound (high intensity) exercises per segment
    compound_counts: Dict[Segment, int] = {seg: 0 for seg in Segment}
    for ex in safe_pool:
        if ex.metabolic_density >= 1 or ex.kinetic_chain_loading == "CLOSED_LOADED":
            compound_counts[ex.primary_segment] += 1
            for sec in ex.secondary_segments:
                compound_counts[sec] += 1

    coverage_detail = []
    for seg in target_segments:
        # Require at least one compound/high-density movement for hypertrophy/strength goals
        is_sufficient = counts[seg] >= min_per_segment
        if goal in ["leg_day", "push_day", "pull_day", "full_body"]:
            if compound_counts[seg] < 1:
                is_sufficient = False
                
        coverage_detail.append(
            SegmentCoverage(
                segment=seg,
                count=counts[seg],
                is_sufficient=is_sufficient,
            )
        )

    missing = [c.segment for c in coverage_detail if not c.is_sufficient]

    if not missing:
        return RefusalDecision(
            should_refuse=False,
            coverage=coverage_detail,
        )

    # Build a human-readable, clinically-grounded refusal message
    missing_names = [seg.value.replace("_", " ").title() for seg in missing]
    goal_label = goal.replace("_", " ").title()

    message = (
        f"Based on the injuries and conditions you reported, a safe and balanced "
        f"{goal_label} workout is not biomechanically possible right now. "
        f"The following muscle groups cannot be adequately trained without risk: "
        f"{', '.join(missing_names)}. "
        "We strongly recommend consulting a physiotherapist who can design a "
        "personalised rehabilitation programme. In the meantime, consider "
        "upper-body active recovery or gentle mobility work if those are safe."
    )

    return RefusalDecision(
        should_refuse=True,
        missing_segments=missing,
        coverage=coverage_detail,
        refusal_message=message,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Goal classifier
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from app.core.config import settings
from app.utils.logger import logger

_GOAL_KEYWORDS: Dict[str, List[str]] = {
    "leg_day":       ["leg day", "leg session", "lower day", "quads", "hamstrings", "legs", "leg workout", "leg plan", "leg training"],
    "lower_body":    ["lower body", "lower-body", "legs and glutes", "lower"],
    "push_day":      ["push day", "push session", "chest day", "chest and shoulders"],
    "pull_day":      ["pull day", "pull session", "back day", "back and biceps"],
    "upper_body":    ["upper body", "upper-body", "upper day"],
    "core":          ["core day", "abs", "core session", "core workout"],
    "full_body":     ["full body", "full-body", "total body", "whole body"],
    "active_recovery": ["active recovery", "recovery day", "light session"],
    "rehab":         ["rehab", "rehabilitation", "physio session"],
}

from typing import Literal

GoalEnum = Literal[
    "leg_day", "lower_body", "push_day", "pull_day", 
    "upper_body", "core", "full_body", "active_recovery", "rehab"
]

class GoalClassification(BaseModel):
    goal: GoalEnum

def classify_goal(user_query: str) -> str:
    """
    LLM-powered intent classification. Maps the user's query to a key in GOAL_SEGMENT_MAP.
    Handles typos, synonyms, and different languages automatically.
    """
    try:
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=settings.OPENAI_API_KEY,
            max_retries=2,
        ).with_structured_output(GoalClassification, method="function_calling")
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a fitness intent classifier. Read the user's workout request and map it to exactly ONE of the following internal goal keys based on which body parts they want to train:\n"
             "- 'leg_day': For lower body, quads, hamstrings, legs, glutes, calves.\n"
             "- 'push_day': For chest, shoulders, triceps, pushing movements.\n"
             "- 'pull_day': For back, biceps, pulling movements, lats.\n"
             "- 'upper_body': For general upper body (chest, back, shoulders, arms).\n"
             "- 'core': For abs, core, obliques, midsection.\n"
             "- 'active_recovery': For stretching, light mobility, recovery.\n"
             "- 'rehab': For physiotherapy, injury rehab.\n"
             "- 'full_body': For full body, general fitness, or if the request is vague/unspecified.\n"
             "Account for typos, slang, and non-English languages. Return ONLY the string key."),
            ("human", "{query}")
        ])
        
        chain = prompt | llm
        result = chain.invoke({"query": user_query})
        valid_keys = list(GOAL_SEGMENT_MAP.keys())
        
        if result.goal in valid_keys:
            logger.info(f"🎯 [Goal Classifier] '{user_query}' -> {result.goal}")
            return result.goal
        else:
            logger.warning(f"⚠️ [Goal Classifier] LLM returned invalid goal '{result.goal}'. Defaulting to full_body.")
            return "full_body"
            
    except Exception as e:
        logger.error(f"❌ [Goal Classifier] LLM failed: {e}. Falling back to keywords.")
        query_lower = user_query.lower()
        for goal, keywords in _GOAL_KEYWORDS.items():
            for kw in keywords:
                if kw in query_lower:
                    return goal
        return "full_body"   # conservative default
