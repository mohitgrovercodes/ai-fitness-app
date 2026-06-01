"""
Biomechanical safety schema for AI Fitness Gym.

Implements 9-feature Biomechanical Tagging (7 original + 2 Phase 2 additions):
  Feature 8: torsional_joint_loading  — distinguishes curtsy lunge from reverse lunge
             for ACL/meniscus patients (rotational stress, not just joint identity).
  Feature 9: spinal_shear_level       — separates anterior shear (RDL, good morning)
             from pure axial compression (back squat) for spondylolisthesis safety.

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
    kinetic_chain_loading: ChainStatus
    axial_compression_level: CompressionLevel
    grip_requirement: GripDemand
    joint_impact_level: ImpactLevel
    upper_limb_stabilization: UpperLimbDemand
    metabolic_density: MetabolicDensity
    # ── Phase 2 additions ──────────────────────────────────────────
    torsional_joint_loading: bool = False
    """
    True when the exercise loads a joint through the rotational/transverse
    plane under significant load (pivoting, crossing, valgus forcing).
    Examples: curtsy lunge, kb_swing (fatigue), russian_twist.
    Distinguishes curtsy lunge from reverse lunge for ACL/meniscus patients.
    """
    spinal_shear_level: ShearLevel = ShearLevel.NONE
    """
    Anterior shear force on the lumbar spine. See ShearLevel for convention.
    An exercise can have LOW axial compression but HIGH shear (RDL).
    An exercise can have HIGH axial compression but LOW shear (back squat).
    """


# ─────────────────────────────────────────────────────────────────────
# Constraint vector (produced by the Intake LLM)
# ─────────────────────────────────────────────────────────────────────
class InjuryConstraint(BaseModel):
    """
    Parallel to the 7 features - one constraint field per feature axis.
    Defaults are the most permissive (HIGH ceilings, empty block lists),
    so an absent constraint = "no restriction on this axis".

    The intake LLM populates this from a raw injury string. Pydantic
    enforces the closed vocabularies at the structured-output layer.
    """
    # Parallel to Feature 1
    blocked_joints: List[Joint] = Field(default_factory=list)
    # Parallel to Feature 2  ← THE MODIFICATION
    # Enables blocking specific chain statuses (e.g. OPEN_UNLOADED for
    # patellofemoral pain) without blocking the joint outright.
    blocked_chains: List[ChainStatus] = Field(default_factory=list)
    # Parallel to Feature 3 (ordinal cap)
    max_axial_compression: CompressionLevel = CompressionLevel.HIGH
    # Parallel to Feature 4 (ordinal cap)
    max_grip_requirement: GripDemand = GripDemand.HEAVY
    # Parallel to Feature 5 (ordinal cap)
    max_impact: ImpactLevel = ImpactLevel.HIGH
    # Parallel to Feature 6 (binary)
    block_upper_limb_active: bool = False
    # Parallel to Feature 7 (ordinal cap)
    max_metabolic_density: MetabolicDensity = MetabolicDensity.HIGH
    # ── Phase 2 additions ──────────────────────────────────────────
    # Parallel to Feature 8 (binary)
    block_torsional_loading: bool = False
    # Parallel to Feature 9 (ordinal cap)
    max_spinal_shear: ShearLevel = ShearLevel.HIGH
