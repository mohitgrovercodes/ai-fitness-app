"""
app/tools/vision_tools.py
=========================
Vision Agent ke liye 3 main helper functions (Tools):

1. search_image_in_db(image_bytes, return_vector=False):
   - PIL Image convert karta hai
   - CLIP model se vector banata hai
   - ChromaDB food_image_centroids se Top-5 matches laata hai
   - return_vector=True: CLIP vector bhi return karta hai (self-learning ke liye)

2. get_food_nutrition(category_name):
   - Category name se food_text DB me search karta hai
   - Exact nutrition data return karta hai

3. identify_and_learn_new_food(image_bytes, clip_vector=None):  ← NEW VLM-based
   - Image ko GPT-4o-mini Vision ko bhejta hai
   - Structured JSON: food_name + calories + protein + carbs + fat
   - ChromaDB text search karta hai (exact name se)
   - Self-Learn: Image vector + Nutrition ChromaDB mein save karta hai
   - Agali baar same dish → CLIP Tier 1 mein pakad lega (FREE!)
"""

import io
import chromadb
import warnings
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from PIL import Image
import torch

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).resolve().parent.parent.parent
CHROMA_DIR = ROOT_DIR / "chromadb_store"

# ─── CLIP Model (lazily loaded once to save startup time) ─────
_clip_model     = None
_clip_processor = None


def _load_clip():
    """Load CLIP model only once into memory (singleton pattern)."""
    global _clip_model, _clip_processor
    if _clip_model is None:
        from transformers import CLIPProcessor, CLIPModel
        print("⏳ [Vision Tool] Loading CLIP model...")
        _clip_model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        _clip_model.eval()
        print("✅ [Vision Tool] CLIP model loaded successfully.")
    return _clip_model, _clip_processor


# ─── ChromaDB Connection (lazily loaded once) ─────────────────
_chroma_client  = None
_food_image_col = None
_food_text_col  = None

from app.core.config import settings
from app.core.database import db_manager
import chromadb.utils.embedding_functions as embedding_functions


def _get_vision_collections():
    """Get collections via Singleton with specific embedding functions."""
    food_image_col = db_manager.get_collection("food_image_centroids")
    
    # Text collection needs OpenAI embedding function
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=settings.OPENAI_API_KEY,
        model_name="text-embedding-3-small"
    )
    food_text_col = db_manager.get_collection("food_text", embedding_function=openai_ef)
    # Note: If collection already exists, setting embedding_function here might not be enough
    # but we will rely on the singleton's PersistentClient.
    return food_image_col, food_text_col


# ════════════════════════════════════════════════════════════════
# TOOL 1: search_image_in_db
# ════════════════════════════════════════════════════════════════
def search_image_in_db(
    image_bytes: bytes,
    return_vector: bool = False,
) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], List[float]]]:
    """
    Converts raw image bytes to a CLIP vector and searches
    ChromaDB food_image_centroids for the Top-5 closest matches.

    Args:
        image_bytes:   Raw PNG/JPG bytes
        return_vector: If True, also returns the normalized CLIP vector
                       (used for self-learning DB writes in Tool 3)

    Returns:
        If return_vector=False: List[Dict] with 'category' and 'score'
        If return_vector=True:  Tuple (List[Dict], List[float])
    """
    clip_model, clip_processor = _load_clip()

    # Step 1: Convert raw bytes → PIL Image
    try:
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise ValueError(f"[Vision Tool] Could not open image bytes: {e}")

    # Step 2: CLIP Vector
    inputs = clip_processor(images=pil_image, return_tensors="pt", padding=True)
    with torch.no_grad():
        vec = clip_model.get_image_features(**inputs)

        if hasattr(vec, "pooler_output"):
            vec = vec.pooler_output
        elif hasattr(vec, "image_embeds"):
            vec = vec.image_embeds
        elif isinstance(vec, dict) and "image_embeds" in vec:
            vec = vec["image_embeds"]

    if not isinstance(vec, torch.Tensor):
        raise ValueError(f"CLIP feature extraction failed. Expected tensor, got {type(vec)}")

    # L2 Normalize (same as stored centroids)
    vec = vec / vec.norm(dim=-1, keepdim=True)
    query_vector = vec.squeeze().numpy().tolist()

    # Step 3: ChromaDB Top-5 Query
    food_image_col, _ = _get_vision_collections()
    results = food_image_col.query(
        query_embeddings=[query_vector],
        n_results=5,
        include=["metadatas", "distances"]
    )

    # Step 4: Distance → Similarity  (similarity = 1 - distance/2)
    top_matches = []
    for i in range(len(results["ids"][0])):
        similarity = round(1 - (results["distances"][0][i] / 2), 4)
        top_matches.append({
            "category": results["metadatas"][0][i].get("category", "unknown"),
            "score":    similarity,
        })

    print(f"🔍 [Vision Tool] Top-5 CLIP Matches: {top_matches}")

    if return_vector:
        return top_matches, query_vector
    return top_matches


# ════════════════════════════════════════════════════════════════
# TOOL 2: get_food_nutrition
# ════════════════════════════════════════════════════════════════
def get_food_nutrition(category_name: str) -> Optional[Dict[str, Any]]:
    """
    Searches food_text ChromaDB using category_name.
    Uses fully dynamic fuzzy string similarity (difflib) to match names —
    no hardcoded word lists. Handles alternate spellings automatically.

    Returns:
        Dict with nutritional info, or None if no good match found.
    """
    import difflib
    _, food_text_col = _get_vision_collections()

    try:
        search_name = category_name.lower().replace("_", " ").strip()

        # Fetch top-N candidates from ChromaDB (count from config)
        results = food_text_col.query(
            query_texts=[category_name],
            n_results=settings.NUTRITION_CANDIDATES_COUNT,
            include=["metadatas", "distances", "documents"]
        )

        if not results["ids"][0]:
            print(f"⚠️ [Vision Tool] No nutrition data found for: {category_name}")
            return None

        # Dynamically pick best match — no hardcoded replacements
        best_result = None
        best_ratio  = 0.0

        for i in range(len(results["ids"][0])):
            metadata       = results["metadatas"][0][i]
            document       = results["documents"][0][i]
            retrieved_name = metadata.get("food_name", "").lower().strip()

            # difflib ratio: 0.0 = completely different, 1.0 = identical
            # Handles: paani/pani, aaloo/aloo, any spelling variant automatically
            ratio = difflib.SequenceMatcher(None, search_name, retrieved_name).ratio()
            print(
                f"   [Vision Tool] Candidate {i+1}: '{retrieved_name}' "
                f"(similarity={ratio:.2f})"
            )
            if ratio > best_ratio:
                best_ratio  = ratio
                best_result = (metadata, document, retrieved_name)

        # Accept match if similarity meets configured threshold
        if best_result and best_ratio >= settings.NUTRITION_SIMILARITY_THRESHOLD:
            metadata, document, retrieved_name = best_result
            print(
                f"✅ [Vision Tool] Nutrition matched: '{search_name}' → "
                f"'{retrieved_name}' (similarity={best_ratio:.2f})"
            )
            return {
                "food_name": metadata.get("food_name", category_name),
                "calories":  metadata.get("calories_kcal", "N/A"),
                "protein":   metadata.get("protein_g",     "N/A"),
                "carbs":     metadata.get("carbs_g",        "N/A"),
                "fat":       metadata.get("fat_g",          "N/A"),
                "raw_text":  document,
            }

        print(
            f"⚠️ [Vision Tool] No match for '{search_name}' "
            f"(best similarity={best_ratio:.2f} < {settings.NUTRITION_SIMILARITY_THRESHOLD})"
        )
        return None

    except Exception as e:
        print(f"❌ [Vision Tool] Error fetching nutrition: {e}")
        return None


# ════════════════════════════════════════════════════════════════
# TOOL 3: identify_and_learn_new_food  ← VLM-based (Upgraded)
# ════════════════════════════════════════════════════════════════
def identify_and_learn_new_food(
    image_bytes: bytes,
    clip_vector: Optional[List[float]] = None,
    clip_hints: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Smart VLM Fallback Tool for OOD (Out-of-Distribution) food images.

    Flow:
      Step 1 → Image ko GPT-4o-mini Vision ko bhejo (with CLIP hints as context)
               → Structured JSON: food_name + nutrition
      Step 2 → Non-Food? → Return {is_food: False, object: ...}
      Step 3 → ChromaDB text lookup (exact name se)
      Step 4 → Self-Learn: Image vector + Nutrition ChromaDB mein save karo
               → Agali baar same dish → CLIP Tier 1 mein pakad lega (FREE!)

    Args:
        image_bytes:  Raw image bytes (PNG/JPG)
        clip_vector:  Optional pre-computed CLIP embedding (for self-learning)
        clip_hints:   Optional list of CLIP's top category guesses (e.g. ['ghevar','anarsa'])
                      Used as context hints for GPT-4o-mini to improve accuracy.

    Returns:
        {
          "is_food":         bool,
          "object":          str | None,
          "identified_food": str | None,
          "source":          "db" | "vlm" | "llm_fallback",
          "nutrition":       dict | None,
          "learned":         bool,
        }
    """
    import base64
    import json
    from openai import OpenAI

    openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)

    # ── Step 1: GPT-4o-mini Vision Call ───────────────────────────────────────
    b64_image = base64.b64encode(image_bytes).decode("utf-8")

    # Build context from CLIP hints so GPT-4o-mini has better context
    hints_context = ""
    if clip_hints:
        clean_hints = [h.replace("_", " ").title() for h in clip_hints[:3]]
        hints_context = (
            f"\n\nContext: Our image recognition system guessed: "
            f"{', '.join(clean_hints)}. "
            f"WARNING: These guesses might be completely wrong due to shape/color confusion. "
            f"DO NOT simply pick one of these hints unless it is a 100% PERFECT visual match (matching shape, texture, AND ingredients). "
            f"If the shape is different (e.g., spherical vs flat patty), IGNORE the hints and provide the true name of the dish (e.g., 'Aloo Bonda', 'Cheese Balls', etc.).\n"
        )

    vlm_prompt = (
        "You are an expert food identification AI. You specialize in Indian and Asian cuisine, but you MUST also identify all global foods (Italian, Mexican, American, Fast Food, etc.).\n"
        "Your task: Identify the EXACT food dish in this image with high accuracy.\n"
        "- If the image contains ANY edible food (including pasta, pizza, tacos, global dishes), you MUST set `is_food: true`.\n"
        "- STRICT CULTURAL POLICY: DO NOT process or analyze BEEF. If the image contains beef, you MUST set `is_food: false` and set the object to 'A dish containing beef (violates policy)'.\n"
        "- Look at the dish's texture, color, ingredients, and presentation carefully.\n"
        "- THALIS / MIXED MEALS: If the image is a 'Thali' or a platter with multiple items, DO NOT just say 'Thali'. You MUST list ONLY the VISIBLE components in the name (e.g., 'North Indian Veg Thali (Dal, Rice, Paneer, Roti)').\n"
        "- CRITICAL ANTI-HALLUCINATION RULE: Do NOT invent or guess traditional combinations. For example, if you see a generic Thali, do NOT assume it contains 'Dal Baati' or 'Churma' unless you specifically see them. List ONLY what your eyes can see.\n"
        "- PORTION ESTIMATION (CRITICAL): You MUST estimate the total quantity of food visible in the image. Do NOT provide 'per 100g' values. Calculate the total calories and macros for the ENTIRE plate/portion shown (e.g. if you see 4 pooris and curry, calculate the sum for all of them).\n"
        "- Do NOT guess. If unsure, pick the most visually accurate name.\n"
        f"{hints_context}\n"
        "Respond in STRICT JSON format only. No other text.\n\n"
        "If the image IS a food dish:\n"
        "{\n"
        '  "is_food": true,\n'
        '  "food_name": "Exact Dish Name (use common English/Hindi name)",\n'
        '  "calories_kcal": <TOTAL number for the entire visible portion>,\n'
        '  "protein_g": <TOTAL number for the entire visible portion. NEVER output N/A. MUST estimate a number>,\n'
        '  "carbs_g": <TOTAL number for the entire visible portion. NEVER output N/A. MUST estimate a number>,\n'
        '  "fat_g": <TOTAL number for the entire visible portion. NEVER output N/A. MUST estimate a number>\n'
        "}\n\n"
        "If the image is NOT food (e.g. person, car, animal, object):\n"
        "{\n"
        '  "is_food": false,\n'
        '  "object": "brief description of what it is"\n'
        "}\n\n"
        "Respond with JSON only. No other text."
    )

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": vlm_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"},
                        },
                    ],
                }
            ],
            max_tokens=120,
            temperature=0.0,
        )

        raw = response.choices[0].message.content.strip()
        # Remove markdown code fences if present
        if "```" in raw:
            raw = raw.split("```")[1].replace("json", "").strip()

        vlm_data = json.loads(raw)
        print(f"🤖 [VLM Tool] GPT-4o-mini response: {vlm_data}")

    except Exception as e:
        print(f"❌ [VLM Tool] GPT-4o-mini call failed: {e}")
        return {
            "is_food":         True,
            "object":          None,
            "identified_food": "unknown food",
            "source":          "llm_fallback",
            "nutrition":       None,
            "learned":         False,
        }

    # ── Step 2: Non-Food Check ─────────────────────────────────────────────────
    if not vlm_data.get("is_food", True):
        object_desc = vlm_data.get("object", "a non-food item")
        print(f"🚫 [VLM Tool] Non-food detected by VLM: '{object_desc}'")
        return {
            "is_food":         False,
            "object":          object_desc,
            "identified_food": None,
            "source":          "vlm",
            "nutrition":       None,
            "learned":         False,
        }

    # ── Step 3: Extract Food Info ──────────────────────────────────────────────
    food_name     = vlm_data.get("food_name", "Unknown Food")
    vlm_nutrition = {
        "food_name": food_name,
        "calories":  vlm_data.get("calories_kcal", "N/A"),
        "protein":   vlm_data.get("protein_g",     "N/A"),
        "carbs":     vlm_data.get("carbs_g",        "N/A"),
        "fat":       vlm_data.get("fat_g",          "N/A"),
    }
    print(f"🍽️ [VLM Tool] Identified: '{food_name}'")

    # ── Step 4: Local DB lookup — enrich macros only, keep VLM calories ──────
    # VLM already analyzed THIS specific plate/portion so its calorie estimate
    # is more accurate than the DB's generic per-100g value.
    # Strategy: Use VLM calories (portion-accurate), fill missing macros from DB.
    db_nutrition = get_food_nutrition(food_name)

    if db_nutrition:
        print(f"✅ [VLM Tool] DB hit for '{food_name}'. Using VLM calories + DB macros.")
        source = "vlm+db"
        def _pick(vlm_val, db_val):
            if vlm_val in (None, "N/A", "n/a", "", 0, "0"):
                return db_val
            return vlm_val
        final_nutrition = {
            "food_name": db_nutrition.get("food_name", food_name),
            "calories":  vlm_nutrition["calories"],
            "protein":   _pick(vlm_nutrition["protein"], db_nutrition.get("protein")),
            "carbs":     _pick(vlm_nutrition["carbs"],   db_nutrition.get("carbs")),
            "fat":       _pick(vlm_nutrition["fat"],     db_nutrition.get("fat")),
        }
    else:
        print(f"🌐 [VLM Tool] DB miss for '{food_name}'. Using VLM nutrition.")
        source = "vlm"
        final_nutrition = vlm_nutrition

    # ── Step 5: Smart Self-Learn — Centroid Update Strategy ───────────────────
    # Rule: Exactly 1 vector per dish (Zero Duplicacy).
    # If dish exists → Moving Average update (old centroid + new image / n+1).
    # If dish is new  → Store directly as first single vector.
    learned = False
    try:
        import numpy as np
        food_image_col, food_text_col = _get_vision_collections()
        dish_key = food_name.lower().replace(" ", "_")

        def _safe_float(val) -> float:
            try:
                return float(val)
            except (TypeError, ValueError):
                return 0.0

        # ── 5a. Image Vector → food_image_centroids ───────────────────────────
        if clip_vector is not None:
            # Check if this dish already has a vector in DB
            existing = food_image_col.get(
                where={"category": dish_key},
                include=["embeddings", "metadatas"]
            )

            if existing["ids"]:
                # ── Dish EXISTS → Moving Average Update ───────────────────────
                existing_id    = existing["ids"][0]
                old_vector     = existing["embeddings"][0]
                old_meta       = existing["metadatas"][0]
                num_images     = int(old_meta.get("num_images", 1))

                # Moving Average: New_Centroid = (Old * n + New) / (n+1)
                old_arr        = np.array(old_vector, dtype=np.float32)
                new_arr        = np.array(clip_vector, dtype=np.float32)
                combined       = (old_arr * num_images + new_arr) / (num_images + 1)

                # L2 Normalize (same as embed_all.py)
                norm           = np.linalg.norm(combined)
                if norm > 0:
                    combined   = combined / norm
                updated_vector = combined.tolist()

                food_image_col.update(
                    ids=[existing_id],
                    embeddings=[updated_vector],
                    metadatas=[{
                        "category":   dish_key,
                        "num_images": num_images + 1,
                        "type":       "vlm_learned_centroid",
                        "source":     "vlm_auto_learned",
                    }],
                )
                print(
                    f"♻️  [VLM Tool] Centroid updated for '{food_name}' "
                    f"(images: {num_images} → {num_images + 1})"
                )
            else:
                # ── Dish NEW → Store as first single vector ────────────────────
                new_id = f"learned_single_{dish_key}"
                food_image_col.add(
                    ids=[new_id],
                    embeddings=[clip_vector],
                    metadatas=[{
                        "category":   dish_key,
                        "num_images": 1,
                        "type":       "vlm_learned_single",
                        "source":     "vlm_auto_learned",
                    }],
                )
                print(f"💾 [VLM Tool] New dish vector saved: '{new_id}'")

        # ── 5b. Nutrition Text → food_text ────────────────────────────────────
        # Save ONLY if dish is brand new (db_nutrition is None).
        # If dish was already in DB → skip (1 dish = 1 text entry, no duplicacy).
        if not db_nutrition:
            nutrition_doc = (
                f"Food: {food_name}\n"
                f"Calories: {vlm_nutrition['calories']} kcal per 100g\n"
                f"Protein: {vlm_nutrition['protein']} g\n"
                f"Carbs: {vlm_nutrition['carbs']} g\n"
                f"Fat: {vlm_nutrition['fat']} g\n"
                f"Source: VLM-identified (auto-learned)"
            )
            text_id = f"learned_text_{dish_key}"
            food_text_col.upsert(
                ids=[text_id],
                documents=[nutrition_doc],
                metadatas=[{
                    "food_name":     food_name,
                    "calories_kcal": _safe_float(vlm_nutrition["calories"]),
                    "protein_g":     _safe_float(vlm_nutrition["protein"]),
                    "carbs_g":       _safe_float(vlm_nutrition["carbs"]),
                    "fat_g":         _safe_float(vlm_nutrition["fat"]),
                    "data_source":   "vlm_auto_learned",
                }],
            )
            print(f"💾 [VLM Tool] Nutrition saved → food_text: '{text_id}'")
        else:
            print(
                f"⏭️  [VLM Tool] Nutrition save skipped — "
                f"'{food_name}' already in DB (no duplicacy)"
            )

        learned = True

    except Exception as e:
        print(f"⚠️ [VLM Tool] Self-learn DB write failed (non-critical): {e}")

    return {
        "is_food":         True,
        "object":          None,
        "identified_food": food_name,
        "source":          source,
        "nutrition":       final_nutrition,
        "learned":         learned,
    }

