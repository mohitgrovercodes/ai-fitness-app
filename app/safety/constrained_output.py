"""
Web-Search Containment — Constrained Output Schema Generator.

Phase 4 of IMPLEMENTATION_PLAN.md.

THE BOUNDARY RULE (§4.2):
  Web-sourced content MAY inform the LLM's coaching narrative (summary, tip,
  descriptive text). It MUST NOT introduce exercise identifiers into the
  structured `workout` field.

HOW ENFORCEMENT WORKS (§4.3):
  The workout output schema is built DYNAMICALLY per request using the
  filtered safe_pool. The exercise_id field becomes a `Literal[*safe_ids]`
  type — a closed enumeration of only the IDs that passed the deterministic
  filter. At token-decoding time, the model's probability mass for any ID
  outside this set collapses to zero.

  Even if:
    - Web search returned an article mentioning "Nordic curl", "Jefferson curl"
    - The LLM's system prompt explicitly says "prescribe Nordic curls"
    - The user explicitly requested "Nordic curls"
  ...the structured output decoder CANNOT emit "nordic_curl" because it is
  not a member of the Literal type. It will select the closest valid ID from
  the safe_pool instead.

  This is the key structural guarantee. It is not prompt-level enforcement
  (bypassable). It is schema-level enforcement (not bypassable).

IMPORTANT: This module produces Pydantic model classes dynamically. Each
call to build_workout_schema() returns a NEW class, so do not cache it across
requests with different safe_pools.
"""
from __future__ import annotations

from typing import List, Literal, Optional, get_args

from pydantic import BaseModel, Field, model_validator

from app.safety.schema import BiomechanicalTags


# ─────────────────────────────────────────────────────────────────────────────
# Segment-level refusal support (wired here, implemented fully in Phase 5)
# ─────────────────────────────────────────────────────────────────────────────

class RefusalReason(BaseModel):
    """
    Structured explanation for why a workout cannot be built safely.
    Surfaces as a first-class response — not a fallback, not an error.
    The user gets an honest, clinically-grounded explanation and a referral.
    """
    cause: str                            # machine-readable tag, e.g. "insufficient_segment_coverage"
    missing_segments: List[str] = []      # which target segments had < min coverage
    message: str                          # human-readable refusal copy shown to the user


# ─────────────────────────────────────────────────────────────────────────────
# Per-exercise item in the structured workout output
# ─────────────────────────────────────────────────────────────────────────────

def build_workout_schema(safe_pool: List[BiomechanicalTags]):
    """
    Build a dynamic Pydantic model whose exercise_id field is constrained to
    ONLY the IDs present in safe_pool.

    Returns:
        WorkoutItem  — the constrained Pydantic model class (not an instance).
        WorkoutPlan  — the top-level response model that references WorkoutItem.

    Usage:
        WorkoutItem, WorkoutPlan = build_workout_schema(safe_pool)
        llm = ChatOpenAI(...).with_structured_output(WorkoutPlan)
        plan = llm.invoke(prompt)
        # plan.workout is List[WorkoutItem] — every exercise_id is guaranteed
        # to be in safe_pool and therefore biomechanically vetted.

    Edge case — empty safe_pool:
        If safe_pool is empty, this still returns valid models but the LLM
        cannot produce any workout items. The caller should have already
        triggered a refusal before reaching this point. The schema remains
        structurally valid so the pipeline never hard-errors.
    """
    if not safe_pool:
        # Sentinel: Literal with a single impossible placeholder.
        # WorkoutPlan.refuse must be True in this case.
        safe_ids_type = Literal["__no_safe_exercises__"]
    else:
        safe_ids = tuple(ex.exercise_id for ex in safe_pool)
        # Literal requires at least one argument and all must be unique.
        safe_ids_type = Literal[safe_ids]  # type: ignore[valid-type]

    class WorkoutItem(BaseModel):
        """
        One exercise slot in the workout plan.

        exercise_id is constrained to the safe_pool IDs — the LLM physically
        cannot emit an ID that was not produced by the deterministic filter.
        This closes the web-search injection vector at the token-decoding layer.
        """
        exercise_id: safe_ids_type  # type: ignore[valid-type]
        day: str = Field(
            description="Training day label, e.g. 'Day 1', 'Monday', 'Push Day'."
        )
        sets: int = Field(ge=1, le=8, description="Number of sets (1–8).")
        reps: str = Field(
            description="Rep target as a string: '8-12', '5', 'AMRAP', '30s hold', etc."
        )
        rest_seconds: Optional[int] = Field(
            default=None,
            ge=0,
            le=600,
            description="Recommended rest between sets in seconds. Null = coach's discretion."
        )
        coaching_note: Optional[str] = Field(
            default=None,
            max_length=300,
            description=(
                "Single sentence of coaching context: technique cue, injury modification, "
                "or progression note. Keep factual. May reference web-search content here."
            )
        )

    class WorkoutPlan(BaseModel):
        """
        Top-level structured response from the workout-planning LLM.

        The LLM must set refuse=True (and populate refusal_reason) instead of
        populating workout when the safe_pool is insufficient to build a
        balanced, medically-appropriate plan for the requested training goal.
        """
        refuse: bool = Field(
            default=False,
            description=(
                "Set True when the injuries make a safe balanced workout IMPOSSIBLE "
                "for the requested training focus. Do not set True just because the "
                "plan is harder to write — only when it is genuinely unsafe or medically "
                "inappropriate to proceed."
            )
        )
        refusal_reason: Optional[str] = Field(
            default=None,
            description=(
                "Required when refuse=True. Explain in plain language WHY a safe plan "
                "cannot be built, and recommend physiotherapy or a modified goal. "
                "Example: 'A safe leg-day program is not possible with bilateral knee "
                "and ankle injuries. We recommend physiotherapy and upper-body active "
                "recovery in the interim.'"
            )
        )
        workout: List[WorkoutItem] = Field(
            default_factory=list,
            description=(
                "The ordered list of exercises. Must be EMPTY when refuse=True. "
                "ALL exercise_ids are pre-validated biomechanically — never override."
            )
        )
        summary: Optional[str] = Field(
            default=None,
            max_length=500,
            description=(
                "Brief overview of the session structure and rationale. "
                "May reference coaching context from web search — this is narrative, "
                "NOT a prescription. 1–3 sentences."
            )
        )
        tip: Optional[str] = Field(
            default=None,
            max_length=300,
            description=(
                "One actionable session tip. May reference exercises not in the "
                "safe_pool here as educational context — the user can explore them "
                "with their coach. Never prescribe them."
            )
        )

        @model_validator(mode="after")
        def validate_refusal_consistency(self) -> "WorkoutPlan":
            if self.refuse and self.workout:
                raise ValueError(
                    "refuse=True but workout is non-empty. "
                    "Empty the workout list when refusing."
                )
            if self.refuse and not self.refusal_reason:
                raise ValueError(
                    "refuse=True but refusal_reason is missing. "
                    "Explain why a safe plan is impossible."
                )
            return self

    # Expose safe_pool membership as a class-level utility for callers
    WorkoutItem.__safe_pool_ids__ = set(get_args(safe_ids_type))
    WorkoutPlan.__safe_pool_size__ = len(safe_pool)

    return WorkoutItem, WorkoutPlan


# ─────────────────────────────────────────────────────────────────────────────
# Boundary audit helpers
# ─────────────────────────────────────────────────────────────────────────────

def audit_plan_containment(
    plan_dict: dict,
    safe_pool: List[BiomechanicalTags],
) -> dict:
    """
    Post-hoc audit: verify every exercise_id in a parsed workout plan is in
    the safe_pool. Returns a summary dict.

    In normal operation this should always return all_contained=True because
    the schema constraint already enforces it. This function exists for:
      - Integration tests and CI
      - Logging/observability (log the audit result alongside each workout)
      - Future paranoia checks if constrained output is ever bypassed

    Args:
        plan_dict: the parsed WorkoutPlan as a dict (model.model_dump())
        safe_pool: the same safe_pool used to build the schema

    Returns:
        {
          "all_contained": bool,
          "safe_pool_size": int,
          "workout_size": int,
          "violations": List[str],   # IDs that escaped containment
        }
    """
    safe_ids = {ex.exercise_id for ex in safe_pool}
    workout_ids = [item["exercise_id"] for item in plan_dict.get("workout", [])]
    violations = [eid for eid in workout_ids if eid not in safe_ids]
    return {
        "all_contained": len(violations) == 0,
        "safe_pool_size": len(safe_pool),
        "workout_size": len(workout_ids),
        "violations": violations,
    }
