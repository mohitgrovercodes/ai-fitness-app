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

Map the injury text to the following constraint dimensions:

1. blocked_joints: Anatomical joints that must not be loaded as a prime mover.
   Use the enums HIP, KNEE, ANKLE, LUMBAR_SPINE, THORACIC_SPINE, CERVICAL_SPINE,
   SHOULDER, ELBOW, WRIST. For "shoulder injury" include SHOULDER. For
   "asthma" or "cardiovascular" issues, do NOT add joints - use max_metabolic_density.

2. blocked_chains: Specific kinetic chain statuses to forbid. CRITICAL: only
   populate this for conditions where chain status matters beyond joint blocking:
   - Patellofemoral pain syndrome / chondromalacia patellae -> block OPEN_UNLOADED
     (the leg-extension machine is the classic culprit, while closed-chain
      squats may actually rehabilitate).
   Leave empty for most other injuries.

3. max_axial_compression: 0=NONE, 1=MEDIUM, 2=HIGH (default). Set to 0 for
   any spinal disc / radiculopathy / acute lumbar injury. Set to 1 for chronic
   managed back issues.

4. max_grip_requirement: 0=NONE, 1=LIGHT, 2=HEAVY (default). Set to 0 for
   acute wrist sprain, severe carpal tunnel, recent hand surgery.

5. max_impact: 0=NONE, 1=LOW, 2=HIGH (default). Set to 0 for any acute
   joint sprain, stress fracture, plantar fasciitis. Set to 1 for chronic
   joint pain that tolerates light tempo work.

6. block_upper_limb_active: true if the user cannot bear bodyweight through
   their arms (acute shoulder injury, post-op upper body, severe wrist sprain).

7. max_metabolic_density: 0=LOW, 1=MEDIUM, 2=HIGH (default). Set to 0 for
   asthma flare, recent cardiac event, advanced pregnancy. Set to 1 for
   moderate CV deconditioning.

Be conservative: when in doubt, lean toward MORE restrictive constraints.
Refusing an unsafe exercise is recoverable; permitting one is not.
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
