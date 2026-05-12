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
        user_context = state.get("user_context", {})
        goal = user_context.get("goal", "General Fitness")
        diet_pref = user_context.get("diet_preference", "None")

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
            from app.core.database import db_manager
            import asyncio
            async with db_manager.lock:
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

        # ── TIER 3: Low CLIP score — VLM Double Check / Auto Reject ───────────────
        if top_score < NON_FOOD_THRESHOLD:
            # SUPER LOW SCORE (< 0.70): Absolute garbage / not food, save cost
            if top_score < 0.70:
                print(f"[Vision Agent] Score ({top_score:.4f} < 0.70). Auto-rejecting to save cost.")
                meta["gpt_vision_used"] = False
                meta["decision_tier"] = f"Guardrail B — Auto-Reject (Score: {top_score:.4f})"
                meta["identified_food"] = "Non-Food Image"
                prompt = self._build_guardrail_b_prompt("something that does not look like food", user_text)
                llm_response = await self.llm.ainvoke(prompt)
                return self._build_output(llm_response.content, meta)
                
            # BORDERLINE SCORE (0.70 - 0.82): Could be an exotic food OR a Car. Let VLM double-check.
            print(f"[Vision Agent] Low CLIP score ({top_score:.4f}). VLM double-check...")
            meta["gpt_vision_used"] = True
            async with db_manager.lock:
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
            meta = self._build_meals_from_result(result, meta)
            prompt = self._build_nutrition_prompt(
                result["identified_food"], result["nutrition"], user_text, result["source"], goal, diet_pref
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

        # ── TIER 1a: High Confidence — Direct Local DB Lookup ─────────────────
        if is_high_confidence and top_score >= CONFIDENCE_THRESHOLD:
            print(f"[Vision Agent] HIGH confidence: '{top_match['category']}' ({top_score:.4f})")
            meta["decision_tier"]   = f"Tier 1a — High Confidence CLIP (score: {top_score:.4f})"
            meta["identified_food"] = top_match["category"]
            async with db_manager.lock:
                nutrition_data = await asyncio.to_thread(get_food_nutrition, top_match["category"])

            # Check if DB macros are complete — if not, trigger VLM for portion-aware estimation
            db_has_macros = nutrition_data and any(
                nutrition_data.get(k, "N/A") != "N/A"
                for k in ["protein", "carbs", "fat"]
            )
            if not db_has_macros:
                print(f"[Vision Agent] Tier 1a: DB macros incomplete for '{top_match['category']}'. Upgrading to VLM.")
                meta["gpt_vision_used"] = True
                meta["decision_tier"]   = f"Tier 1a+VLM — CLIP Confident + VLM Nutrition ({top_score:.4f})"
                async with db_manager.lock:
                    result = await asyncio.to_thread(
                        identify_and_learn_new_food, image_bytes, clip_vector=clip_vector, clip_hints=clip_hints
                    )
                if result.get("is_food") and result.get("nutrition"):
                    nutrition_data = result["nutrition"]
                    meta["data_source"] = result.get("source", "vlm")
                else:
                    meta["data_source"] = "llm_knowledge"
            else:
                meta["data_source"] = "db"

            result_for_meals = {
                "identified_food": top_match["category"],
                "nutrition": nutrition_data,
                "source": meta["data_source"],
                "learned": False,
                "is_food": True,
            }
            meta = self._build_meals_from_result(result_for_meals, meta)
            prompt = self._build_nutrition_prompt(
                top_match["category"], nutrition_data, user_text, meta["data_source"], goal, diet_pref
            )

        else:
            # ── TIER 2: Ambiguous or OOD → VLM Portion Estimation ───────────────
            reason = "Ambiguous Food" if not is_high_confidence else "OOD Food"
            print(f"[Vision Agent] {reason} (score: {top_score:.4f}). Triggering VLM for portion estimation...")
            meta["gpt_vision_used"] = True
            meta["decision_tier"]   = f"Tier 2 — {reason} (CLIP: {top_score:.4f}) → VLM Portion Estimation"

            async with db_manager.lock:
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

            meta = self._build_meals_from_result(result, meta)
            prompt = self._build_nutrition_prompt(
                result["identified_food"], result["nutrition"], user_text, result["source"], goal, diet_pref
            )

        # ══════════════════════════════════════════════════════
        # STAGE 3 — LLM REASONING & FINAL RESPONSE
        # ══════════════════════════════════════════════════════
        print("[Vision Agent] Sending to LLM for final reasoning...")
        llm_response = await self.llm.ainvoke(prompt)
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
        goal: str = "General Fitness",
        diet_pref: str = "None"
    ) -> str:

        """Universal nutrition prompt — works for Tier 1a (DB) and VLM fallback."""
        food_display = food_name.replace("_", " ").title()
        source_note  = (
            "" if source == "db"
            else "\n\n*(Nutritional values sourced via AI analysis and may be approximate.)*"
        )

        if nutrition:
            serving_note = f"Estimated Serving Size: ~{nutrition['estimated_serving']}\n" if nutrition.get("estimated_serving") else ""
            nutrition_context = (
                f"Food: {nutrition.get('food_name', food_display)}\n"
                f"{serving_note}"
                f"Calories: {nutrition.get('calories', 'N/A')} kcal\n"
                f"Protein: {nutrition.get('protein', 'N/A')} g\n"
                f"Carbs:   {nutrition.get('carbs', 'N/A')} g\n"
                f"Fat:     {nutrition.get('fat', 'N/A')} g\n\n"
                f"CRITICAL INSTRUCTION: If any macro value is 'N/A', or if calories seem wildly inaccurate for a standard moderate serving (e.g. >1000 kcal for a normal home-cooked plate), you MUST ignore that data and estimate realistic numeric values for the given portion. All values must be numbers — never 'N/A'."
            )
        else:
            nutrition_context = (
                f"Food: {food_display}\n"
                f"(No database entry found. You MUST estimate all values based on a typical moderate serving size of this dish.)"
            )

        if user_text and user_text.strip():
            instruction = (
                f'The user asked: "{user_text}"\n'
                f"Answer using the nutrition data above. FOLLOW THIS FORMAT STRICTLY:\n\n"
                f"## 🍽️ [Food Name]\n"
                f"**Description**: [2-3 sentences — interactive, engaging, describe what the dish is, its origin, key ingredients, and taste profile.]\n\n"
                f"## 📊 Nutritional Breakdown\n"
                f"*(Estimated for the visible portion)*\n"
                f"- **Serving Size**: [Xg or approximate]\n"
                f"- **Calories**: [X] kcal\n"
                f"- **Protein**: [X] g\n"
                f"- **Carbohydrates**: [X] g\n"
                f"- **Fat**: [X] g\n\n"
                f"STRICT RULES: All values MUST be numeric. Never write 'N/A' or 'Generally high'. "
                f"Do NOT add a 'Complementary Aspects', 'Tips', or any other section. End after the nutritional breakdown."
            )
        else:
            instruction = (
                "Provide a response using EXACTLY this format:\n\n"
                "## 🍽️ [Food Name]\n"
                "**Description**: [2-3 sentences — interactive, engaging, describe what the dish is, its origin, key ingredients, and taste profile.]\n\n"
                "## 📊 Nutritional Breakdown\n"
                "*(Estimated for the visible portion)*\n"
                "- **Serving Size**: [Xg or approximate]\n"
                "- **Calories**: [X] kcal\n"
                "- **Protein**: [X] g\n"
                "- **Carbohydrates**: [X] g\n"
                "- **Fat**: [X] g\n\n"
                "STRICT RULES: All values MUST be numeric. Never write 'N/A' or 'Generally high'. "
                "Do NOT add a 'Complementary Aspects', 'Tips', 'Putting it Together', or any other section. End immediately after the nutritional breakdown."
            )

        return (
            f"You are FitBot, a world-class food identification and nutrition AI.\n"
            f"You can identify and provide nutrition for ANY food from ANY global cuisine.\n\n"
            f"USER PROFILE:\n"
            f"- Goal: {goal}\n"
            f"- Dietary Preference: {diet_pref}\n\n"
            f"You have ALREADY analyzed the user's image using a vision module. The exact nutritional data for their specific meal is provided below.\n\n"
            f"{nutrition_context}\n\n"
            f"{instruction}\n\n"
            f"CRITICAL RULE: DO NOT say 'I cannot analyze this specific meal' or 'I cannot see the image'. You ALREADY analyzed it. Present the provided data confidently as your own analysis.\n"
            f"Use clean markdown headers. Do not add any 'expert coach' fluff or unrelated suggestions.{source_note}"
        )

    def _build_meals_from_result(self, result: dict, meta: dict) -> dict:
        """
        Universally populate meta['meals'] from any VLM result dict.
        Uses 4-4-9 macro math fallback if any macro is N/A or missing.
        Called from ALL CLIP decision paths so meals are always populated.
        """
        nut = result.get("nutrition")
        if not nut:
            return meta
        try:
            def _to_num(val):
                if val in (None, "N/A", "n/a", "", 0, "0"):
                    return None
                try:
                    return float(str(val).replace("g", "").strip())
                except Exception:
                    return None

            cal  = _to_num(nut.get("calories")) or 0.0
            prot = _to_num(nut.get("protein"))
            fat  = _to_num(nut.get("fat"))
            carb = _to_num(nut.get("carbs"))

            # 4-4-9 rule fallback for any missing macro
            if prot is None and fat is not None and carb is not None:
                prot = max(0, round((cal - (fat * 9) - (carb * 4)) / 4, 1))
            elif fat is None and prot is not None and carb is not None:
                fat  = max(0, round((cal - (prot * 4) - (carb * 4)) / 9, 1))
            elif carb is None and prot is not None and fat is not None:
                carb = max(0, round((cal - (prot * 4) - (fat * 9)) / 4, 1))
            elif prot is None and fat is None and carb is None and cal > 0:
                # All missing — balanced 40/30/30 split estimate
                prot = round(cal * 0.30 / 4, 1)
                fat  = round(cal * 0.30 / 9, 1)
                carb = round(cal * 0.40 / 4, 1)

            food_name = nut.get("food_name", result.get("identified_food", "Food"))
            meta["meals"] = [{
                "type": "Analyzed Image",
                "name": str(food_name).title(),
                "calories": round(cal, 1),
                "protein": f"{prot}g" if prot is not None else "Est. N/A",
                "carbs": f"{carb}g" if carb is not None else "Est. N/A",
                "fat": f"{fat}g" if fat is not None else "Est. N/A",
                "benefit": "Analyzed from your uploaded image."
            }]
        except Exception as e:
            print(f"[Vision Agent] _build_meals_from_result failed: {e}")
        return meta

    def _build_output(self, response_text: str, meta: dict = None) -> Dict[str, Any]:
        """Standard output format for the LangGraph state with optional metadata block."""
        full_response = response_text
        if meta:
            full_response = response_text + "\n\n" + self._format_metadata(meta)
        
        out = {
            "answer": full_response,
            "status": "success",
            "metadata": meta
        }
        if meta and "meals" in meta:
            out["meals"] = meta["meals"]
            
        return {
            "messages": [AIMessage(content=full_response)],
            "specialist_results": {
                "vision": out
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