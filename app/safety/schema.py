"""
Biomechanical safety schema for AI Fitness Gym.

Implements 10-feature Biomechanical Tagging:
  Feature 8:  torsional_joint_loading  — distinguishes curtsy lunge from reverse lunge
              for ACL/meniscus patients (rotational stress, not just joint identity).
  Feature 9:  spinal_shear_level       — separates anterior shear (RDL, good morning)
              from pure axial compression (back squat) for spondylolisthesis safety.
  Feature 10: joint_actions (Phase 3)  — encodes WHAT a joint is doing, not just WHICH
              joint is involved. Fixes the PFPS bug: KNEE_EXTENSION_OPEN blocks
              leg_extension but leaves KNEE_FLEXION_OPEN (leg curl) perfectly safe.
              blocked_chains on InjuryConstraint is deprecated in favour of this.

All enums are closed vocabularies. Ordinal features use IntEnum so the
deterministic filter can compare with `<=`. Categorical features use
str Enum and are checked with set membership.

Closed-vocabulary design means the intake LLM cannot hallucinate a body
region or constraint type - Pydantic validation rejects unknown values
at the JSON-schema layer.
"""
from enum import Enum, IntEnum
from typing import List, Optional

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────
# Feature 1: Joints (set-valued, categorical)
# ─────────────────────────────────────────────────────────────────────
class Joint(str, Enum):
    """
    Anatomical joints that act as PRIME MOVERS during the exercise.

    Definition: a joint that undergoes dynamic flexion/extension/rotation
    UNDER LOAD as the primary movement. Stabilizing joints that stay
    isometric (e.g. spine during a deadlift) are NOT included here -
    spinal risk for those is captured via `axial_compression_level`.
    """
    HIP = "HIP"
    KNEE = "KNEE"
    ANKLE = "ANKLE"
    LUMBAR_SPINE = "LUMBAR_SPINE"        # dynamic flexion/extension (crunch, hyperextension)
    THORACIC_SPINE = "THORACIC_SPINE"    # rotation (russian twist)
    CERVICAL_SPINE = "CERVICAL_SPINE"
    SHOULDER = "SHOULDER"
    ELBOW = "ELBOW"
    WRIST = "WRIST"


# ─────────────────────────────────────────────────────────────────────
# Feature 2: Kinetic chain status (categorical, single value)
# ─────────────────────────────────────────────────────────────────────
class ChainStatus(str, Enum):
    """How the moving limb relates to a fixed surface."""
    CLOSED_LOADED = "CLOSED_LOADED"        # foot/hand fixed against resistance (squat, push-up)
    CLOSED_SUPPORTED = "CLOSED_SUPPORTED"  # limb fixed but body weight supported (leg press, supine work)
    OPEN_UNLOADED = "OPEN_UNLOADED"        # limb moves freely (leg extension, lateral raise)


# ─────────────────────────────────────────────────────────────────────
# Feature 3: Axial compression (ordinal)
# ─────────────────────────────────────────────────────────────────────
class CompressionLevel(IntEnum):
    """Compressive force acting down the spine. Ordinal: NONE < MEDIUM < HIGH."""
    NONE = 0    # supine/prone, machine-supported, bodyweight horizontal
    MEDIUM = 1  # dumbbells held at sides, goblet position, light load
    HIGH = 2    # barbell on back/shoulders/chest, conventional deadlifts


# ─────────────────────────────────────────────────────────────────────
# Feature 9: Spinal shear (ordinal)  ← Phase 2 addition
# ─────────────────────────────────────────────────────────────────────
class ShearLevel(IntEnum):
    """
    Anterior shear force on the lumbar spine from long-moment-arm hinges.
    Distinct from axial compression: a back squat compresses vertically
    (HIGH compression) but has LOW shear. An RDL has MODERATE compression
    but HIGH shear due to the horizontal moment arm.

    Clinically relevant for: spondylolisthesis, spondylolysis, disc
    herniation with anterior instability.

    Ordinal: NONE < MEDIUM < HIGH.
    """
    NONE = 0    # vertical-load or supine/seated — squat, calf raise, plank, crunch
    MEDIUM = 1  # moderate forward lean — sumo deadlift, back squat, hyperextension
    HIGH = 2    # long horizontal moment arm — RDL, good morning, conventional deadlift


# ─────────────────────────────────────────────────────────────────────
# Feature 10: Joint action vocabulary  ← Phase 3 addition
# ─────────────────────────────────────────────────────────────────────
class JointAction(str, Enum):
    """
    Encodes the specific movement a joint performs during an exercise,
    including both the ACTION (flexion/extension/rotation) and the
    CHAIN STATUS (open/closed).

    This replaces the blunt `blocked_chains` approach on InjuryConstraint.
    Example: PFPS constraint = blocked_joint_actions: [KNEE_EXTENSION_OPEN_CHAIN]
      - leg_extension  [KNEE_EXTENSION_OPEN_CHAIN]  → BLOCKED  ✓
      - lying_leg_curl [KNEE_FLEXION_OPEN_CHAIN]    → ALLOWED  ✓  (was BLOCKED in v0)
      - back_squat     [KNEE_EXTENSION_CLOSED_CHAIN] → ALLOWED  ✓
    """
    # ── Knee ─────────────────────────────────────────────────────────
    KNEE_EXTENSION_OPEN   = "KNEE_EXTENSION_OPEN_CHAIN"    # leg extension machine
    KNEE_EXTENSION_CLOSED = "KNEE_EXTENSION_CLOSED_CHAIN"  # squat ascent, step-up
    KNEE_FLEXION_OPEN     = "KNEE_FLEXION_OPEN_CHAIN"      # lying/seated leg curl
    KNEE_FLEXION_CLOSED   = "KNEE_FLEXION_CLOSED_CHAIN"    # squat descent
    # ── Hip ──────────────────────────────────────────────────────────
    HIP_FLEXION_OPEN      = "HIP_FLEXION_OPEN_CHAIN"       # lying leg raise, hanging leg raise
    HIP_FLEXION_CLOSED    = "HIP_FLEXION_CLOSED_CHAIN"     # squat descent, hinge loading
    HIP_EXTENSION_CLOSED  = "HIP_EXTENSION_CLOSED_CHAIN"   # squat ascent, deadlift, hip thrust
    HIP_ABDUCTION_OPEN    = "HIP_ABDUCTION_OPEN_CHAIN"     # clamshell, side-lying abduction
    # ── Spine ────────────────────────────────────────────────────────
    SPINAL_FLEXION_DYNAMIC    = "SPINAL_FLEXION_DYNAMIC"       # crunch, hanging leg raise
    SPINAL_EXTENSION_DYNAMIC  = "SPINAL_EXTENSION_DYNAMIC"     # hyperextension
    SPINAL_ROTATION_DYNAMIC   = "SPINAL_ROTATION_DYNAMIC"      # russian twist
    SPINAL_ISOMETRIC_BRACING  = "SPINAL_ISOMETRIC_BRACING"     # plank, bird-dog, dead bug
    SPINAL_LATERAL_FLEXION    = "SPINAL_LATERAL_FLEXION"        # side plank (isometric lateral)
    # ── Shoulder ─────────────────────────────────────────────────────
    SHOULDER_OVERHEAD_LOADED      = "SHOULDER_OVERHEAD_LOADED"      # overhead press
    SHOULDER_HORIZONTAL_LOADED    = "SHOULDER_HORIZONTAL_LOADED"    # bird-dog arm, pallof press
    # ── Elbow ────────────────────────────────────────────────────────
    ELBOW_FLEXION_OPEN    = "ELBOW_FLEXION_OPEN_CHAIN"     # bicep curl
    ELBOW_FLEXION_CLOSED  = "ELBOW_FLEXION_CLOSED_CHAIN"   # pull-up, row
    ELBOW_EXTENSION_OPEN  = "ELBOW_EXTENSION_OPEN_CHAIN"   # tricep extension
    ELBOW_EXTENSION_CLOSED= "ELBOW_EXTENSION_CLOSED_CHAIN" # push-up, bench press
    # ── Wrist ────────────────────────────────────────────────────────
    WRIST_FLEXION_OPEN    = "WRIST_FLEXION_OPEN_CHAIN"     # wrist curl
    WRIST_EXTENSION_OPEN  = "WRIST_EXTENSION_OPEN_CHAIN"   # reverse wrist curl
    # ── Ankle ────────────────────────────────────────────────────────
    ANKLE_PLANTARFLEXION_LOADED   = "ANKLE_PLANTARFLEXION_LOADED"   # calf raise


# ─────────────────────────────────────────────────────────────────────
# Feature 4: Grip demand (ordinal)
# ─────────────────────────────────────────────────────────────────────
class GripDemand(IntEnum):
    """Grip force required. Ordinal: NONE < LIGHT < HEAVY."""
    NONE = 0    # bodyweight or machine-supported, no hand contact with load
    LIGHT = 1   # dumbbell held, banded, light/moderate cable
    HEAVY = 2   # deadlift, kettlebell swing, hanging exercises (sustained grip)


# ─────────────────────────────────────────────────────────────────────
# Feature 5: Impact (ordinal)
# ─────────────────────────────────────────────────────────────────────
class ImpactLevel(IntEnum):
    """Ballistic/plyometric joint impact. Ordinal: NONE < LOW < HIGH."""
    NONE = 0   # controlled tempo, isometrics, machines
    LOW = 1    # jogging, light tempo, light ballistic (KB swing)
    HIGH = 2   # box jumps, sprints, depth jumps, plyometric variants


# ─────────────────────────────────────────────────────────────────────
# Feature 6: Upper-limb support (binary)
# ─────────────────────────────────────────────────────────────────────
class UpperLimbDemand(str, Enum):
    """Whether the upper limbs actively support body weight."""
    NONE = "NONE"     # most lower body lifts, supine/seated work
    ACTIVE = "ACTIVE"  # planks, push-ups, hanging exercises, bird-dog


# ─────────────────────────────────────────────────────────────────────
# Feature 7: Metabolic density (ordinal)
# ─────────────────────────────────────────────────────────────────────
class MetabolicDensity(IntEnum):
    """CV/respiratory demand. Ordinal: LOW < MEDIUM < HIGH."""
    LOW = 0      # heavy strength with long rest, isolation, mobility
    MEDIUM = 1   # compound lifts with moderate rest, supersets
    HIGH = 2     # circuits, sprints, AMRAPs, KB complexes


# ─────────────────────────────────────────────────────────────────────
# Per-exercise tag bundle
# ─────────────────────────────────────────────────────────────────────
class BiomechanicalTags(BaseModel):
    """The full safety tag set attached to every exercise in the library."""
    exercise_id: str
    name: str
    primary_joints_involved: List[Joint]
    kinetic_chain_loading: ChainStatus      # kept for debug/display; safety logic uses joint_actions
    axial_compression_level: CompressionLevel
    grip_requirement: GripDemand
    joint_impact_level: ImpactLevel
    upper_limb_stabilization: UpperLimbDemand
    metabolic_density: MetabolicDensity
    # ── Phase 2 additions ──────────────────────────────────────────
    torsional_joint_loading: bool = False
    spinal_shear_level: ShearLevel = ShearLevel.NONE
    # ── Phase 3 addition ───────────────────────────────────────────
    joint_actions: List[JointAction] = Field(
        default_factory=list,
        description=(
            "The specific joint movements performed under load during this exercise. "
            "Encodes both action-type (flexion/extension) and chain status (open/closed). "
            "Used by the filter to match blocked_joint_actions on InjuryConstraint."
        )
    )


# ─────────────────────────────────────────────────────────────────────
# Constraint vector (produced by the Intake LLM)
# ─────────────────────────────────────────────────────────────────────
class InjuryConstraint(BaseModel):
    """
    One constraint field per feature axis. Defaults are maximally permissive
    (HIGH ceilings, empty block lists) — absent constraint = no restriction.

    The intake LLM populates this from a raw injury string. Pydantic
    enforces closed vocabularies at the structured-output layer.
    """
    # Feature 1: block specific joints (anatomical identity)
    blocked_joints: List[Joint] = Field(default_factory=list)

    # Feature 3: axial compression ceiling
    max_axial_compression: CompressionLevel = CompressionLevel.HIGH
    # Feature 4: grip ceiling
    max_grip_requirement: GripDemand = GripDemand.HEAVY
    # Feature 5: impact ceiling
    max_impact: ImpactLevel = ImpactLevel.HIGH
    # Feature 6: upper-limb active support flag
    block_upper_limb_active: bool = False
    # Feature 7: metabolic density ceiling
    max_metabolic_density: MetabolicDensity = Field(default=MetabolicDensity.HIGH)
    block_torsional_loading: bool = Field(default=False)
    max_spinal_shear: ShearLevel = Field(default=ShearLevel.HIGH)

    # Feature 10: exact joint-action blocklist  ← Phase 3
    blocked_joint_actions: List[JointAction] = Field(
        default_factory=list,
        description=(
            "Block specific joint movements (action + chain status). "
            "More precise than blocked_chains: KNEE_EXTENSION_OPEN blocks leg_extension "
            "but leaves KNEE_FLEXION_OPEN (leg curl) safely available."
        )
    )

class TriageReasoning(BaseModel):
    """
    Chain-of-Thought wrapper for InjuryConstraint.
    Forces the LLM to write out its clinical analysis before committing to
    the structured constraint vector.
    """
    clinical_analysis: str = Field(
        description=(
            "Step-by-step clinical analysis of the user's injury. "
            "Consider: What anatomy is affected? Is weight-bearing through arms or legs compromised? "
            "What specific movements must be avoided? Does it affect systemic cardiovascular capacity?"
        )
    )
    constraint: InjuryConstraint = Field(
        description="The final deterministic constraint vector derived directly from the clinical_analysis."
    )

