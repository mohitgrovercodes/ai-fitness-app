"""
Deterministic safety filter - the absolute physical safety shield.

Operates entirely on structured tags. No strings, no LLM calls, no
fuzzy matching. Given a list of tagged exercise candidates and an
InjuryConstraint vector, returns the subset that passes ALL ten
biomechanical safety checks.

Each check corresponds to exactly one of the 10 features. Failures are
attributed to a specific axis (auditable, debuggable).
"""
from typing import List, Tuple

from app.safety.schema import (
    BiomechanicalTags,
    InjuryConstraint,
    UpperLimbDemand,
    ShearLevel,
    JointAction,
)


def safety_violations(
    ex: BiomechanicalTags, c: InjuryConstraint
) -> List[str]:
    """
    Return a list of axis names that this exercise violates. Empty list
    means the exercise is safe under the given constraint. Useful for
    debugging which rule blocked a specific exercise.
    """
    violations: List[str] = []

    # 1. Joint loading
    if set(ex.primary_joints_involved) & set(c.blocked_joints):
        violations.append("blocked_joint")

    # 3. Axial compression (ordinal)
    if ex.axial_compression_level > c.max_axial_compression:
        violations.append("axial_compression_exceeded")
    # 4. Grip (ordinal)
    if ex.grip_requirement > c.max_grip_requirement:
        violations.append("grip_exceeded")
    # 5. Impact (ordinal)
    if ex.joint_impact_level > c.max_impact:
        violations.append("impact_exceeded")
    # 6. Upper-limb stabilization
    if c.block_upper_limb_active and ex.upper_limb_stabilization == UpperLimbDemand.ACTIVE:
        violations.append("upper_limb_loading_forbidden")
    # 7. Metabolic density (ordinal)
    if ex.metabolic_density > c.max_metabolic_density:
        violations.append("metabolic_density_exceeded")
    # 8. Torsional / rotational joint loading  ← Phase 2
    if c.block_torsional_loading and ex.torsional_joint_loading:
        violations.append("torsional_loading_forbidden")
    # 9. Spinal shear (ordinal)  ← Phase 2
    if ex.spinal_shear_level > c.max_spinal_shear:
        violations.append("spinal_shear_exceeded")
    # 10. Joint-action blocklist (set intersection)  ← Phase 3
    # More precise than check 2 (chain status). Blocks KNEE_EXTENSION_OPEN
    # without touching KNEE_FLEXION_OPEN — the PFPS rehab distinction.
    if c.blocked_joint_actions and set(ex.joint_actions) & set(c.blocked_joint_actions):
        violations.append("blocked_joint_action")

    return violations


def is_safe(ex: BiomechanicalTags, c: InjuryConstraint) -> bool:
    """Boolean fast-path - True if no axis violated."""
    return not safety_violations(ex, c)


def filter_safe_exercises(
    candidates: List[BiomechanicalTags],
    constraint: InjuryConstraint,
) -> List[BiomechanicalTags]:
    """Return the subset of candidates that pass all 7 safety axes."""
    return [ex for ex in candidates if is_safe(ex, constraint)]


def filter_with_audit(
    candidates: List[BiomechanicalTags],
    constraint: InjuryConstraint,
) -> Tuple[List[BiomechanicalTags], List[Tuple[BiomechanicalTags, List[str]]]]:
    """
    Two-output variant for the evaluation harness:
      - safe: passing exercises
      - blocked: tuples of (exercise, violations) explaining each rejection
    """
    safe: List[BiomechanicalTags] = []
    blocked: List[Tuple[BiomechanicalTags, List[str]]] = []
    for ex in candidates:
        v = safety_violations(ex, constraint)
        if not v:
            safe.append(ex)
        else:
            blocked.append((ex, v))
    return safe, blocked
