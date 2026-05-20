from typing import Dict, Any, List, Optional, Type
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from app.core.state import AgentState
from app.utils.logger import logger

from app.core.config import settings

# Activity level → Harris-Benedict / Mifflin-St Jeor multiplier table.
# Stored here once so it is not duplicated across agents.
_ACTIVITY_MULTIPLIERS = {
    "SEDENTARY":         1.2,
    "LIGHTLY_ACTIVE":    1.375,
    "MODERATELY_ACTIVE": 1.55,
    "VERY_ACTIVE":       1.725,
    "EXTRA_ACTIVE":      1.9,
}


def _compute_tdee(weight_kg, height_cm, age, gender: str, activity_level: str) -> dict:
    """
    Mifflin-St Jeor BMR → TDEE.  Pure arithmetic — no string pattern matching.
    Returns a dict with bmr, tdee, cal_loss (−20 %), cal_maintenance, cal_gain (+15 %).
    All values are 0 when the profile is incomplete.
    """
    try:
        w = float(weight_kg)
        h = float(height_cm)
        a = float(age)
        if w <= 0 or h <= 0 or a <= 0:
            raise ValueError
    except (ValueError, TypeError):
        return {"bmr": 0, "tdee": 0, "cal_loss": 0, "cal_maintenance": 0, "cal_gain": 0}

    if str(gender).upper() in ("MALE", "M"):
        bmr = (10 * w) + (6.25 * h) - (5 * a) + 5
    else:
        bmr = (10 * w) + (6.25 * h) - (5 * a) - 161

    factor = _ACTIVITY_MULTIPLIERS.get(str(activity_level).upper(), 1.2)
    tdee   = bmr * factor

    return {
        "bmr":             round(bmr),
        "tdee":            round(tdee),
        "cal_loss":        round(max(bmr, tdee * 0.80)),   # 20 % deficit, never below BMR
        "cal_maintenance": round(tdee),
        "cal_gain":        round(tdee * 1.15),             # 15 % surplus
    }


class BaseRAGAgent:
    """
    Base class for Specialist Agents using Adaptive RAG / CRAG.
    Deduplicates the Phase 1 (DB), Phase 2 (Multi-Query), and Phase 3 (Web) logic.
    """
    def __init__(
        self, 
        agent_name: str,
        rag_tool: Any, 
        web_search_tool: Any, 
        output_schema: Type[BaseModel],
        system_prompt: str,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.3
    ):
        self.agent_name = agent_name
        self.rag_tool = rag_tool
        self.web_search = web_search_tool
        self.llm = ChatOpenAI(
            model=model_name, 
            temperature=temperature, 
            api_key=settings.OPENAI_API_KEY,
            max_retries=3 # Production-grade retry logic
        ).with_structured_output(output_schema, method="function_calling")
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "CONVERSATION SUMMARY (for context):\n{summary}\n\nQUESTION: {query}\n\nRETRIEVED DATA:\n{context}")
        ])

    async def run_logic(self, state: AgentState, specialist_key: str, topic: str = "general") -> Dict[str, Any]:
        """Core Adaptive RAG logic shared across specialists."""
        query        = state['messages'][-1].content
        user_context = state.get('user_context', {}) or {}
        summary      = state.get('conversation_summary', "No previous context.")

        # ── Extract full onboarding profile ──────────────────────────────────
        goal           = user_context.get("goal", "General Fitness") or "General Fitness"
        diet_pref      = user_context.get("diet_preference") or "None"
        injuries_list  = user_context.get("injuries", []) or []
        injuries       = ", ".join(str(i) for i in injuries_list) if injuries_list else "None"
        med_list       = user_context.get("medical_conditions", []) or []
        medical        = ", ".join(str(m) for m in med_list) if med_list else "None"
        full_name      = user_context.get("full_name") or "User"
        age            = user_context.get("age") or "Unknown"
        gender         = user_context.get("gender") or "Unknown"
        weight_kg      = user_context.get("weight_kg") or "Unknown"
        height_cm      = user_context.get("height_cm") or "Unknown"
        activity_level = user_context.get("activity_level") or "Unknown"

        # ── Compute TDEE inline (pure arithmetic, no goal-string matching) ───
        tdee = _compute_tdee(weight_kg, height_cm, age, gender, activity_level)

        if tdee["tdee"]:
            tdee_str = (
                f"BMR {tdee['bmr']} kcal | TDEE {tdee['tdee']} kcal/day\n"
                f"  Weight-loss target  : {tdee['cal_loss']} kcal  (−20 % deficit, ≥ BMR)\n"
                f"  Maintenance target  : {tdee['cal_maintenance']} kcal\n"
                f"  Weight-gain target  : {tdee['cal_gain']} kcal  (+15 % surplus)\n"
                f"  → Choose the target that matches the user's goal above."
            )
        else:
            tdee_str = "Unknown — profile data incomplete (age / weight / height missing)."

        # ── Build Intelligence Context (Layer 2) ─────────────────────────────
        # Inject computed biometrics so LLM can make goal-appropriate decisions
        # WITHOUT any hardcoded if-else rules in our code.
        intelligence_context = self._build_intelligence_context(
            weight_kg, goal, tdee, diet_pref, injuries, medical
        )

        logger.info(
            f"🧬 [{self.agent_name}] Profile: name='{full_name}', goal='{goal}', "
            f"diet='{diet_pref}', {weight_kg}kg, TDEE={tdee.get('tdee', '?')} kcal"
        )

        chain = self.prompt | self.llm

        # ── PHASE 1: Profile-Enriched RAG Search (Layer 1) ───────────────────
        # Enrich the search query with user profile so ChromaDB embeddings
        # return goal-relevant results — no extra LLM call, zero extra cost.
        enriched_query = self._build_enriched_query(query, goal, diet_pref, weight_kg, injuries)
        db_results  = await self.rag_tool.search(enriched_query, diet_preference=diet_pref)
        context_str = self._format_context(db_results)

        # Shared prompt variables — full profile + intelligence context passed to every agent
        prompt_vars = {
            "query":                query,
            "context":              context_str or "No specific data retrieved from local database.",
            "goal":                 goal,
            "injuries":             injuries,
            "medical":              medical,
            "diet_preference":      diet_pref,
            "summary":              summary,
            "full_name":            full_name,
            "age":                  str(age),
            "gender":               str(gender),
            "weight_kg":            str(weight_kg),
            "height_cm":            str(height_cm),
            "activity_level":       str(activity_level),
            "tdee":                 tdee_str,
            "intelligence_context": intelligence_context,
        }

        analysis = await chain.ainvoke(prompt_vars)

        logger.info(f"[{self.agent_name}] Accurate: {analysis.is_accurate} | Web needed: {analysis.needs_web_search}")

        # ── PHASE 2: Multi-Query Expansion ───────────────────────────────────
        if not analysis.is_accurate and hasattr(analysis, 'sub_queries') and analysis.sub_queries:
            logger.info(f"  🔄 [{self.agent_name}] Expanding search with sub-queries...")

            multiplier  = getattr(analysis, 'quantity_multiplier', 1.0)
            db_results  = await self.rag_tool.multi_query_search(query, analysis.sub_queries, multiplier=multiplier, diet_preference=diet_pref)
            context_str = self._format_context(db_results)

            prompt_vars["context"] = context_str or "Expanded search returned no additional data."
            analysis = await chain.ainvoke(prompt_vars)

        # ── PHASE 3: Web Fallback ─────────────────────────────────────────────
        if not analysis.is_accurate and analysis.needs_web_search:
            if self.web_search.is_available:
                logger.info(f"  🌐 [{self.agent_name}] Triggering Web Search fallback...")
                web_data    = await self.web_search.search(query, topic=topic)
                web_context = web_data.get("summary", "") or \
                              "\n".join([r['content'] for r in web_data.get("results", [])[:3]])

                prompt_vars["context"] = f"{context_str}\n\n[Live Web Data]:\n{web_context}"
                analysis = await chain.ainvoke(prompt_vars)
            else:
                logger.warning(f"  ⚠️ [{self.agent_name}] Web search unavailable. Falling back to expert knowledge.")
                prompt_vars["context"] = f"{context_str}\n\n[Expert Knowledge Fallback]: Use expert knowledge to supplement the data."
                analysis = await chain.ainvoke(prompt_vars)

        # Prepare final specialist output
        specialist_output = {
            "answer": analysis.final_answer,
            "status": "success" if analysis.is_accurate else "expert_knowledge"
        }
        
        # Add optional fields if present in the analysis object
        # This dynamically captures metadata like 'exercise_gifs', 'nutritional_info', etc.
        logger.info(f"[{self.agent_name}] Raw Analysis: {analysis}")
        standard_fields = {"is_accurate", "needs_web_search", "sub_queries", "final_answer", "quantity_multiplier"}
        analysis_dict = analysis.model_dump() if hasattr(analysis, "model_dump") else analysis.__dict__
        # Media fields that should always be included (even if empty dict)
        always_include = {"exercise_gifs", "exercise_images"}
        
        for field, val in analysis_dict.items():
            if field not in standard_fields:
                # Always include media fields; only skip other falsy values
                if val or field in always_include:
                    specialist_output[field] = val

        # Run optional validation hook for subclasses
        specialist_output = self._validate_output(specialist_output, context_str, state)

        return {
            "specialist_results": {
                specialist_key: specialist_output
            }
        }

    def _validate_output(self, output: Dict[str, Any], context: str, state: AgentState = None) -> Dict[str, Any]:
        """Optional hook for subclasses to validate or format output."""
        return output

    def _format_context(self, results: List[Dict]) -> str:
        """To be implemented by subclasses if they need custom formatting."""
        raise NotImplementedError("Subclasses must implement _format_context")

    @staticmethod
    def _build_enriched_query(query: str, goal: str, diet: str, weight_kg, injuries: str) -> str:
        """
        Layer 1: Profile-enriched RAG search query.
        Combines user message with their profile so ChromaDB embedding search
        returns goal-relevant results. No hardcoded keywords — uses actual user data.
        Zero extra LLM calls.
        """
        parts = [query.strip()]
        if goal and goal.lower() not in ("none", "unknown", "general fitness"):
            parts.append(f"goal: {goal}")
        if diet and diet.lower() not in ("none", "unknown"):
            parts.append(f"diet: {diet}")
        if weight_kg and str(weight_kg).lower() not in ("none", "unknown"):
            parts.append(f"body weight: {weight_kg}kg")
        if injuries and injuries.lower() != "none":
            parts.append(f"injuries: {injuries}")
        enriched = ". ".join(parts)
        return enriched

    @staticmethod
    def _build_intelligence_context(weight_kg, goal: str, tdee: dict, diet: str, injuries: str, medical: str) -> str:
        """
        Layer 2: Intelligence Block.
        Computes and formats physiologically grounded targets from user biometrics.
        The LLM uses its OWN expertise to apply these numbers to the user's goal —
        no hardcoded if-else goal logic here. Zero extra cost.
        """
        try:
            w = float(weight_kg)
        except (ValueError, TypeError):
            w = 0.0

        # Physiology-based protein range (universal, not goal-specific)
        # 1.6g/kg = minimum for active individuals (ISSN consensus)
        # 2.2g/kg = upper practical limit for most goals
        if w > 0:
            protein_min = round(w * 1.6, 1)
            protein_max = round(w * 2.2, 1)
            protein_line = f"Protein Target Range: {protein_min}g – {protein_max}g/day (based on {w}kg body weight)"
        else:
            protein_line = "Protein Target: Use standard guidelines (body weight unavailable)"

        # TDEE-based calorie context
        if tdee.get("tdee"):
            cal_line = (
                f"Calorie Targets: loss={tdee['cal_loss']} kcal | "
                f"maintenance={tdee['cal_maintenance']} kcal | "
                f"gain={tdee['cal_gain']} kcal"
            )
        else:
            cal_line = "Calorie Targets: unavailable (incomplete profile)"

        return (
            f"INTELLIGENCE CONTEXT (Use your expertise to apply this to the user's goal: '{goal}'):\n"
            f"  {protein_line}\n"
            f"  {cal_line}\n"
            f"  Diet Constraint: {diet}\n"
            f"  Injury/Medical Constraints: {injuries} | {medical}\n"
            f"  → Determine the most appropriate exercise type, rep ranges, macro split, "
            f"    and food priorities based on the goal above. Do NOT use generic defaults."
        )
