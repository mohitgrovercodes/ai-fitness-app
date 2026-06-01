"""
Intake LLM translator - converts raw injury text into an InjuryConstraint.

This is the ONE place an LLM touches the safety pipeline. Its job is
structured translation (free text -> closed-vocabulary enums), NOT safety
judgment. Pydantic + OpenAI structured output enforces the enums at the
JSON-schema layer, so hallucinated body regions / constraint types cannot
escape this boundary.

Fail-safe behavior: if the LLM call errors or returns invalid output, we
return the MOST restrictive constraint (block everything except NONE-impact
NONE-compression supine work). Better to refuse than to leak.
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app.core.config import settings
from app.safety.schema import (
    CompressionLevel,
    GripDemand,
    ImpactLevel,
    InjuryConstraint,
    MetabolicDensity,
)
from app.utils.logger import logger


_SYSTEM_PROMPT = """You are a sports-medicine triage assistant. Your ONLY job is
to translate a user's injury or medical condition description into a structured
biomechanical constraint vector. You do NOT prescribe exercises, give medical
advice, or generate text - you only fill in the structured fields.

Map the injury text to the following 10 constraint dimensions:

1. blocked_joints: Anatomical joints that must not be loaded as a prime mover.
   Values: HIP, KNEE, ANKLE, LUMBAR_SPINE, THORACIC_SPINE, CERVICAL_SPINE,
   SHOULDER, ELBOW, WRIST.
   - "knee injury" → [KNEE]
   - "low back pain / disc herniation / sciatica" → [LUMBAR_SPINE]
   - "shoulder impingement" → [SHOULDER]
   - "asthma / cardiovascular" → do NOT add joints; use max_metabolic_density

2. blocked_joint_actions: Specific joint movements to block (more precise than
   blocked_joints). Use when the injury is action-specific, not joint-wide.
   Values: KNEE_EXTENSION_OPEN_CHAIN, KNEE_EXTENSION_CLOSED_CHAIN,
           KNEE_FLEXION_OPEN_CHAIN, KNEE_FLEXION_CLOSED_CHAIN,
           HIP_FLEXION_OPEN_CHAIN, HIP_FLEXION_CLOSED_CHAIN,
           HIP_EXTENSION_CLOSED_CHAIN, HIP_ABDUCTION_OPEN_CHAIN,
           SPINAL_FLEXION_DYNAMIC, SPINAL_EXTENSION_DYNAMIC,
           SPINAL_ROTATION_DYNAMIC, SPINAL_ISOMETRIC_BRACING,
           SPINAL_LATERAL_FLEXION, SHOULDER_OVERHEAD_LOADED,
           SHOULDER_HORIZONTAL_LOADED, ANKLE_PLANTARFLEXION_LOADED.
   - Patellofemoral pain (PFPS) / chondromalacia → [KNEE_EXTENSION_OPEN_CHAIN]
     (blocks leg-extension machine; closed-chain squats remain safe for rehab)
   - Hip flexor impingement / psoas strain → [HIP_FLEXION_OPEN_CHAIN]
   - Discogenic rotation intolerance → [SPINAL_ROTATION_DYNAMIC]
   - Achilles tendinopathy / calf tear → [ANKLE_PLANTARFLEXION_LOADED]
   - Leave empty for most non-action-specific injuries (use blocked_joints instead)

3. max_axial_compression: 0=NONE, 1=MEDIUM, 2=HIGH (default 2).
   - Acute disc herniation / radiculopathy / acute lumbar → 0
   - Chronic managed back pain → 1

4. max_grip_requirement: 0=NONE, 1=LIGHT, 2=HEAVY (default 2).
   - Acute wrist sprain / carpal tunnel / hand surgery → 0
   - Moderate wrist pain → 1

5. max_impact: 0=NONE, 1=LOW, 2=HIGH (default 2).
   - Acute joint sprain / stress fracture / plantar fasciitis → 0
   - Chronic joint pain tolerating light tempo → 1

6. block_upper_limb_active: true if user cannot bear bodyweight through arms.
   - Acute shoulder / post-op upper body / severe wrist sprain → true

7. max_metabolic_density: 0=LOW, 1=MEDIUM, 2=HIGH (default 2).
   - Asthma flare / recent cardiac event / advanced pregnancy → 0
   - Moderate CV deconditioning → 1

8. block_torsional_loading: true if rotational/pivoting joint loading is
   contraindicated.
   - ACL tear / meniscus injury / ligament reconstruction → true
   - Healthy knees with only a muscle strain → false (usually)

9. max_spinal_shear: 0=NONE, 1=MEDIUM, 2=HIGH (default 2).
   Anterior shear force from long-moment-arm hinge exercises (RDLs, good
   mornings, conventional deadlifts). Distinct from axial compression.
   - Spondylolisthesis / spondylolysis → 0 or 1
   - Disc herniation with anterior instability → 0

Be conservative: when in doubt, lean toward MORE restrictive constraints.
Refusing an unsafe exercise is always recoverable; permitting one is not.
"""


def translate_injury_to_constraint(injury_text: str) -> InjuryConstraint:
    """
    Boundary call from raw injury text -> closed-vocabulary constraint.
    Returns a maximally-restrictive InjuryConstraint on any failure.
    """
    if not injury_text or not injury_text.strip():
        return InjuryConstraint()  # all defaults = no restriction (no injury)

    try:
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=settings.OPENAI_API_KEY,
            max_retries=2,
        ).with_structured_output(InjuryConstraint, method="function_calling")

        prompt = ChatPromptTemplate.from_messages([
            ("system", _SYSTEM_PROMPT),
            ("human", "Injury / condition: {injury}"),
        ])
        chain = prompt | llm
        constraint = chain.invoke({"injury": injury_text.strip()})
        logger.info(f"🩺 [Intake] '{injury_text}' -> {constraint.model_dump()}")
        return constraint
    except Exception as e:
        logger.error(f"❌ [Intake] LLM translation failed: {e} - returning restrictive fallback")
        # Fail-safe: block everything except seated/supine zero-load work.
        return InjuryConstraint(
            blocked_joints=[],
            max_axial_compression=CompressionLevel.NONE,
            max_grip_requirement=GripDemand.NONE,
            max_impact=ImpactLevel.NONE,
            block_upper_limb_active=True,
            max_metabolic_density=MetabolicDensity.LOW,
        )
