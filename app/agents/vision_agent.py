"""
app/agents/vision_agent.py
===========================
7.1 VISION AGENT — Self-Learning + Safety Guardrails

Decision Engine:
  Guardrail A  — Text-only / Off-topic query → Polite redirect
  Tier 3       — Non-Food Reject  (CLIP score < NON_FOOD_THRESHOLD, VLM confirmed)
  Tier 2 / 1b — OOD / Ambiguous  → VLM Fallback + Self-Learn DB
  Guardrail B  — Non-Food image detected by VLM → Context-aware reject
  Tier 1a      — High Confidence  (score >= CONFIDENCE_THRESHOLD, gap >= 0.01)
"""

from typing import Dict, Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage

from app.core.state import AgentState
from app.core.config import settings
from app.tools.vision_tools import (
    search_image_in_db,
    get_food_nutrition,
    identify_and_learn_new_food,
)

# ─── Constants (from central config) ──────────────────────────────────────────
CONFIDENCE_THRESHOLD = settings.VISION_CONFIDENCE_THRESHOLD   # 0.86
AMBIGUITY_GAP        = settings.AMBIGUITY_GAP_THRESHOLD        # 0.01
NON_FOOD_THRESHOLD   = settings.VISION_NON_FOOD_THRESHOLD      # 0.82




class VisionAgent:
    """
    Step 7.1 — VISION AGENT (Food Only)
    Processes uploaded food images using:
    - CLIP for fast local matching (Tier 1)
    - GPT-4o-mini Vision for OOD identification + Self-Learning (Tier 2)
    - Safety Guardrails for non-food images and off-topic queries
    """

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model=model_name, temperature=0.3)

    async def run(self, state: AgentState) -> Dict[str, Any]:
        """
        Main entry point for the Vision Agent node in the LangGraph workflow.
        """
        print("[Vision Agent] Starting food image analysis...")

        # ── Extract Inputs ─────────────────────────────────────────────────────
        image_bytes = state.get("image_bytes")
        user_text   = state["messages"][-1].content if state.get("messages") else ""

        # ── Metadata Tracker ───────────────────────────────────────────────────
        meta = {
            "clip_used":         False,
            "clip_top_match":    None,
            "clip_top_score":    None,
            "clip_top_5":        None,
            "gpt_vision_used":   False,
            "identified_food":   None,
            "decision_tier":     None,
            "data_source":       None,
            "self_learned":      False,
        }

        # ══════════════════════════════════════════════════════
        # GUARDRAIL A: Text-only / Off-topic Query
        # ══════════════════════════════════════════════════════
        if not image_bytes:
            is_food_related = await self._is_food_related_query(user_text)
            if not is_food_related:
                print("[Vision Agent] GUARDRAIL A: Text-only off-topic query.")
                meta["decision_tier"] = "Guardrail A — Off-topic Text"
                return self._build_output(self._guardrail_a_message(), meta)
            meta["decision_tier"] = "Guardrail A — Food Query, No Image"
            return self._build_output(self._ask_for_image_message(user_text), meta)

        # ══════════════════════════════════════════════════════
        # PRE-FLIGHT: Image Quality Check (before ANY AI call)
        # ══════════════════════════════════════════════════════
        quality_issue = self._check_image_quality(image_bytes)
        if quality_issue:
            print(f"[Vision Agent] PRE-FLIGHT rejected: {quality_issue}")
            meta["decision_tier"] = f"Pre-Flight Reject — {quality_issue}"
            return self._build_output(
                f"⚠️ I couldn't process this image — it appears to be **{quality_issue}**.\n\n"
                "Please upload a **clear, well-lit photo** of your food item and I'll analyse it instantly! 📸",
                meta
            )

        # ══════════════════════════════════════════════════════
        # STAGE 1 — CLIP Visual Search (Top-5)
        # ══════════════════════════════════════════════════════
        try:
            from app.core.database import db_singleton
            import asyncio
            async with db_singleton.lock:
                top_matches, clip_vector = await asyncio.to_thread(
                    search_image_in_db, image_bytes, return_vector=True
                )
        except Exception as e:
            print(f"[Vision Agent] CLIP search failed: {e}")
            meta["decision_tier"] = "Error — CLIP Failed"
            return self._build_output(
                "I had trouble processing your image. "
                "Please make sure it is a clear food photo and try again.",
                meta
            )

        top_match = top_matches[0]
        top_score = top_match["score"]
        clip_hints = [m["category"] for m in top_matches[:3]]

        # Update metadata with CLIP results
        meta["clip_used"]      = True
        meta["clip_top_match"] = top_match["category"]
        meta["clip_top_score"] = top_score
        meta["clip_top_5"]     = top_matches

        print(f"[Vision Tool] Top-5 CLIP Matches: {top_matches}")

        # ══════════════════════════════════════════════════════
        # STAGE 2 — DECISION ENGINE
        # ══════════════════════════════════════════════════════

        # ── TIER 3: Low CLIP score — VLM double-check ─────────────────────────
        if top_score < NON_FOOD_THRESHOLD:
            print(f"[Vision Agent] Low CLIP score ({top_score:.4f}). VLM double-check...")
            meta["gpt_vision_used"] = True
            async with db_singleton.lock:
                result = await asyncio.to_thread(
                    identify_and_learn_new_food, image_bytes, clip_vector=None, clip_hints=clip_hints
                )
            if not result["is_food"]:
                print(f"[Vision Agent] GUARDRAIL B (Tier 3): non-food: '{result['object']}'")
                meta["decision_tier"] = f"Guardrail B — Not Food ({result['object']})"
                meta["identified_food"] = result["object"]
                prompt = self._build_guardrail_b_prompt(result["object"], user_text)
                llm_response = await self.llm.ainvoke(prompt)
                return self._build_output(llm_response.content, meta)
            meta["identified_food"] = result["identified_food"]
            meta["data_source"]     = result["source"]
            meta["self_learned"]    = result["learned"]
            prompt = self._build_nutrition_prompt(
                result["identified_food"], result["nutrition"], user_text, result["source"]
            )
            llm_response = await self.llm.ainvoke(prompt)
            return self._build_output(llm_response.content, meta)

        # ── TIER 1b: Ambiguity Check ─────────────────────────────────────────
        is_high_confidence = True
        if len(top_matches) >= 2 and top_score >= CONFIDENCE_THRESHOLD:
            gap   = top_score - top_matches[1]["score"]
            name1 = top_match["category"].lower().replace("_", " ")
            name2 = top_matches[1]["category"].lower().replace("_", " ")
            if gap < AMBIGUITY_GAP:
                if not (name1 in name2 or name2 in name1):
                    print(f"[Vision Agent] True Ambiguity: '{name1}' vs '{name2}' (gap: {gap:.4f})")
                    is_high_confidence = False
                else:
                    print(f"[Vision Agent] Ambiguity canceled: same family.")

        # ── TIER 2 & 1b: OOD or AMBIGUOUS → VLM FALLBACK + SELF-LEARN ─────────
        if top_score < CONFIDENCE_THRESHOLD or not is_high_confidence:
            reason = "OOD Food" if top_score < CONFIDENCE_THRESHOLD else "Ambiguous Food"
            print(f"[Vision Agent] {reason} (score: {top_score:.4f}). Triggering VLM...")
            meta["gpt_vision_used"] = True
            meta["decision_tier"]   = f"Tier 2 — {reason} (CLIP: {top_score:.4f}) → VLM Fallback"

            async with db_singleton.lock:
                result = await asyncio.to_thread(
                    identify_and_learn_new_food, image_bytes, clip_vector=clip_vector, clip_hints=clip_hints
                )

            # GUARDRAIL B — VLM says NOT FOOD
            if not result["is_food"]:
                print(f"[Vision Agent] GUARDRAIL B: not food: '{result['object']}'")
                meta["decision_tier"]   = f"Guardrail B — Not Food ({result['object']})"
                meta["identified_food"] = result["object"]
                prompt = self._build_guardrail_b_prompt(result["object"], user_text)
                llm_response = await self.llm.ainvoke(prompt)
                return self._build_output(llm_response.content, meta)

            meta["identified_food"] = result["identified_food"]
            meta["data_source"]     = result["source"]
            meta["self_learned"]    = result["learned"]
            learned_msg = " 💾 Saved to DB!" if result.get("learned") else ""
            print(f"[Vision Agent] VLM identified: '{result['identified_food']}'{learned_msg}")

            prompt = self._build_nutrition_prompt(
                result["identified_food"], result["nutrition"], user_text, result["source"]
            )
            llm_response = await self.llm.ainvoke(prompt)
            return self._build_output(llm_response.content, meta)

        # ── TIER 1a: High Confidence — Direct Local DB Lookup ──────────────────
        print(f"[Vision Agent] HIGH confidence: '{top_match['category']}' ({top_score:.4f})")
        meta["decision_tier"]   = f"Tier 1a — High Confidence CLIP (score: {top_score:.4f})"
        meta["identified_food"] = top_match["category"]
        async with db_singleton.lock:
            nutrition_data = await asyncio.to_thread(get_food_nutrition, top_match["category"])
        meta["data_source"] = "db" if nutrition_data else "llm_knowledge"
        prompt = self._build_nutrition_prompt(
            top_match["category"], nutrition_data, user_text, "db"
        )

        # ══════════════════════════════════════════════════════
        # STAGE 3 — LLM REASONING & FINAL RESPONSE
        # ══════════════════════════════════════════════════════
        print("[Vision Agent] Sending to LLM for final reasoning...")
        llm_response  = await self.llm.ainvoke(prompt)
        print("[Vision Agent] Response ready.")
        return self._build_output(llm_response.content, meta)

    # ─── Image Quality Pre-Flight ─────────────────────────────────────────────

    @staticmethod
    def _check_image_quality(image_bytes: bytes) -> str | None:
        """
        Fast pre-flight check BEFORE any AI (CLIP/GPT) is invoked.
        Returns a string describing the problem, or None if image is OK.

        Catches:
          - Corrupt / unreadable files
          - Too small files (< 1 KB  — likely blank/icon)
          - Nearly black images  (avg brightness < 15)
          - Nearly white images  (avg brightness > 245)
          - Very small resolution (< 64×64 px)
        """
        import io as _io
        from PIL import Image as _Image, ImageStat as _Stat

        # 1. File size check — anything < 1 KB is almost certainly bad
        if len(image_bytes) < 1024:
            return "too small or empty (< 1 KB)"

        # 2. Try to open the image
        try:
            img = _Image.open(_io.BytesIO(image_bytes)).convert("RGB")
        except Exception:
            return "corrupt or unreadable file"

        # 3. Resolution check
        w, h = img.size
        if w < 64 or h < 64:
            return f"too low resolution ({w}×{h} px)"

        # 4. Brightness check — nearly black or nearly white
        stat      = _Stat.Stat(img)
        avg_brightness = sum(stat.mean) / 3          # average of R, G, B channels
        if avg_brightness < 15:
            return "nearly black / silhouette image"
        if avg_brightness > 245:
            return "nearly white / blank image"

        return None   # ✅ Image looks fine

    # ─── Prompt Builders ──────────────────────────────────────────────────────

    async def _is_food_related_query(self, user_text: str) -> bool:
        """
        Dynamically classifies whether the user's text is related to food,
        nutrition, diet, or meal analysis — using gpt-4o-mini.
        Works for any language (English, Hindi, Hinglish, etc.).
        Returns True if food-related, False otherwise.
        """
        if not user_text or not user_text.strip():
            # Empty text with no image → not food related
            return False

        prompt = (
            "You are a query classifier for a Food Vision AI assistant.\n"
            "Determine if the user's message is related to food, nutrition, diet, "
            "meal analysis, calories, health, or food images.\n"
            "Also return True if the user seems to be uploading/referring to an image (even without food context).\n\n"
            f"User message: \"{user_text}\"\n\n"
            "Reply with ONLY one word: 'yes' if food/nutrition/image related, 'no' otherwise."
        )
        try:
            response = await self.llm.ainvoke(prompt)
            answer   = response.content.strip().lower()
            is_food  = answer.startswith("yes")
            print(f"[Vision Agent] Guardrail A classifier → '{answer}' (food_related={is_food})")
            return is_food
        except Exception as e:
            print(f"[Vision Agent] Guardrail A classifier failed: {e}. Defaulting to food-related=True.")
            # Safe fallback: assume food-related so we don't block valid queries
            return True

    def _guardrail_a_message(self) -> str:
        """Guardrail A: Off-topic text query with no image."""
        return (
            "Hi there! 👋 I'm **FitBot's Food Vision Agent**.\n\n"
            "I specialize in **analyzing food images** and providing detailed nutritional information. "
            "I'm not able to help with other topics.\n\n"
            "📸 **Please upload a clear photo of your meal or food item**, and I'll give you:\n"
            "- 🍽️ The exact dish name\n"
            "- 🔥 Calories & macros (Protein, Carbs, Fat)\n"
            "- 💡 A personalized health tip\n\n"
            "Looking forward to helping you eat smarter! 😊"
        )

    def _ask_for_image_message(self, user_text: str) -> str:
        """User asked about food but didn't upload an image."""
        return (
            "Hi! 👋 It looks like you're asking about food nutrition, but I don't see any image attached.\n\n"
            "📸 **Please upload a clear photo of your food**, and I'll instantly identify it and give you "
            "the full nutritional breakdown!\n\n"
            "I can identify thousands of Indian and international dishes — just send the photo! 🍽️"
        )

    def _build_guardrail_b_prompt(self, object_desc: str, user_text: str) -> str:
        """Guardrail B: Non-food image detected by VLM."""
        return (
            f"You are FitBot, a friendly AI Food Vision Agent.\n\n"
            f"A user uploaded an image, but it appears to be: **{object_desc}** (not a food item).\n\n"
            f"Respond warmly and helpfully:\n"
            f"1. Acknowledge what the image appears to show (e.g., '{object_desc}').\n"
            f"2. Explain that you are a Food Vision Agent specialized only in food images.\n"
            f"3. Politely ask them to upload a clear photo of a meal or food item instead.\n"
            f"4. Keep it friendly, short (3-4 sentences), and encouraging.\n\n"
            f"Do NOT be dismissive. Be warm and guide them to upload a food photo."
        )

    def _build_nutrition_prompt(
        self,
        food_name: str,
        nutrition: Dict | None,
        user_text: str,
        source: str,
    ) -> str:
        """Universal nutrition prompt — works for Tier 1a (DB) and VLM fallback."""
        food_display = food_name.replace("_", " ").title()
        source_note  = (
            "" if source == "db"
            else "\n\n*(Nutritional values sourced via AI analysis and may be approximate.)*"
        )

        if nutrition:
            nutrition_context = (
                f"Food: {nutrition.get('food_name', food_display)}\n"
                f"Calories: {nutrition.get('calories', 'N/A')} kcal\n"
                f"Protein: {nutrition.get('protein', 'N/A')} g\n"
                f"Carbs:   {nutrition.get('carbs', 'N/A')} g\n"
                f"Fat:     {nutrition.get('fat', 'N/A')} g"
            )
        else:
            nutrition_context = (
                f"Food: {food_display}\n"
                f"(Exact data not available — use your best nutritional knowledge.)"
            )

        if user_text and user_text.strip():
            instruction = (
                f'The user asked: "{user_text}"\n'
                f"Answer their specific question using the nutrition data above."
            )
        else:
            instruction = (
                "The user did not ask a specific question. "
                "Automatically provide a friendly, complete nutritional summary of this food "
                "including what it is, its calories, macros, and a brief health tip."
            )

        return (
            f"You are FitBot, an expert AI nutrition assistant.\n\n"
            f"A user uploaded a food image. You have identified the food as:\n\n"
            f"{nutrition_context}\n\n"
            f"{instruction}\n\n"
            f"Be concise, friendly, and formatted clearly with markdown headers.{source_note}"
        )

    def _build_output(self, response_text: str, meta: dict = None) -> Dict[str, Any]:
        """Standard output format for the LangGraph state with optional metadata block."""
        full_response = response_text
        if meta:
            full_response = response_text + "\n\n" + self._format_metadata(meta)
        return {
            "messages": [AIMessage(content=full_response)],
            "specialist_results": {
                "vision": full_response
            }
        }

    def _format_metadata(self, meta: dict) -> str:
        """Formats a clean diagnostic metadata block appended to every response."""
        clip_match_str = (
            f"`{meta['clip_top_match']}` (score: {meta['clip_top_score']:.4f})"
            if meta.get("clip_top_match") else "N/A"
        )
        food_display = (
            meta["identified_food"].replace("_", " ").title()
            if meta.get("identified_food") else "N/A"
        )
        source_map = {
            "db":            "🗄️ Local Database (Exact Match)",
            "vlm":           "🌐 AI Analysis (Approximate)",
            "llm_knowledge": "🧠 LLM General Knowledge",
            "llm_fallback":  "🧠 LLM Fallback (API Error)",
            None:            "N/A",
        }
        data_src     = source_map.get(meta.get("data_source"), meta.get("data_source", "N/A"))
        decision     = meta.get("decision_tier", "N/A")

        # Pre-compute booleans → plain strings (avoid backslash-in-f-string Python <3.12)
        clip_ok      = "✅ Yes" if meta.get("clip_used")        else "❌ No"
        gpt_ok       = "✅ Yes" if meta.get("gpt_vision_used")  else "❌ No — CLIP was confident"
        learned_ok   = "✅ Yes — Future queries FREE!" if meta.get("self_learned") else "❌ No"

        clip_top5_str = ""
        if meta.get("clip_top_5"):
            rows = [
                f"  {i+1}. `{m['category']}` — {m['score']:.4f}"
                for i, m in enumerate(meta["clip_top_5"])
            ]
            clip_top5_str = "\n**🔍 CLIP Top-5 Matches:**\n" + "\n".join(rows)

        lines = [
            "\n---",
            "### 📊 Vision Analysis Details",
            "| Property | Value |",
            "|---|---|",
            f"| 🍽️ **Identified Food**        | {food_display}  |",
            f"| 🚦 **Decision Tier**           | {decision}      |",
            f"| 🔍 **CLIP Used**               | {clip_ok}        |",
            f"| 🎯 **CLIP Top Match**          | {clip_match_str} |",
            f"| 🤖 **GPT-4o-mini Vision Used** | {gpt_ok}         |",
            f"| 🗄️ **Nutrition Source**        | {data_src}       |",
            f"| 💾 **Self-Learned (DB)**       | {learned_ok}     |",
        ]
        if clip_top5_str:
            lines.append(clip_top5_str)
        return "\n".join(lines)