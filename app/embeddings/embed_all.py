"""
============================================================
  FIT BOT — One-Time Embedding Pipeline
  Run once. Everything stored in ChromaDB permanently.

  What this script does:
    1. FOOD IMAGE CENTROIDS  (CLIP, local — no API cost)
       - Reads all images in each of the 16 category folders
       - Generates a CLIP vector per image
       - Averages all vectors → 1 "perfect centroid" per category
       - Stores in ChromaDB collection: food_image_centroids

    2. FOOD TEXT  (OpenAI text-embedding-3-small)
       - Reads final_master_food.json  (~39K items)
       - Builds a descriptive text string per item
       - Sends to OpenAI in batches of 100
       - Stores in ChromaDB collection: food_text

    3. EXERCISE TEXT  (OpenAI text-embedding-3-small)
       - Reads final_master_exercises.json  (~2.8K items)
       - Builds a descriptive text string per item
       - Sends to OpenAI in batches of 100
       - Stores in ChromaDB collection: exercise_text

  NOTE: Exercise GIFs / images are NOT embedded.
        They are retrieved by ID from the JSON at query time.
============================================================
"""

import os
import json
import time
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

# ── Load environment ──────────────────────────────────────
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not found in .env file!")

# ── Paths ─────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent
DATA_DIR        = BASE_DIR / "Data"
FOOD_JSON       = DATA_DIR / "final_master_food.json"
EXERCISE_JSON   = DATA_DIR / "final_master_exercises.json"
FOOD_IMG_DIR    = DATA_DIR / "Food-dataset" / "food" / "Dataset"
CHROMA_DIR      = BASE_DIR / "chromadb_store"

# ── Imports ───────────────────────────────────────────────
import chromadb
from openai import OpenAI

print("=" * 60)
print("  FIT BOT — Embedding Pipeline Starting")
print("=" * 60)

# ─────────────────────────────────────────────────────────
# PHASE 1: FOOD IMAGE CENTROIDS  (CLIP — runs locally)
# ─────────────────────────────────────────────────────────
print("\n[PHASE 1] Food Image Centroids via CLIP")
print("-" * 40)

from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

print("  → Loading CLIP model (first run downloads ~600MB)...")
clip_model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()
print("  ✓ CLIP model loaded")

def get_clip_vector(image_path: Path) -> np.ndarray | None:
    """Return a normalised CLIP image embedding as a numpy array."""
    try:
        img    = Image.open(image_path).convert("RGB")
        inputs = clip_processor(images=img, return_tensors="pt", padding=True)
        with torch.no_grad():
            vec = clip_model.get_image_features(**inputs)
        vec = vec / vec.norm(dim=-1, keepdim=True)   # L2 normalise
        return vec.squeeze().numpy().tolist()
    except Exception as e:
        print(f"    ⚠ Skipping {image_path.name}: {e}")
        return None


# Connect to ChromaDB
chroma_client   = chromadb.PersistentClient(path=str(CHROMA_DIR))

# Delete old collection if re-running so we start fresh
try:
    chroma_client.delete_collection("food_image_centroids")
    print("  → Cleared existing food_image_centroids collection")
except Exception:
    pass

food_img_collection = chroma_client.get_or_create_collection(
    name="food_image_centroids",
    metadata={"hnsw:space": "cosine"}
)

# Discover all 16 category folders
category_folders = sorted([f for f in FOOD_IMG_DIR.iterdir() if f.is_dir()])
print(f"  → Found {len(category_folders)} food categories: "
      f"{[f.name for f in category_folders]}")

centroids_added = 0
for folder in category_folders:
    category_name = folder.name          # e.g. "samosa"
    image_files   = list(folder.glob("*.jpg")) + list(folder.glob("*.jpeg")) + \
                    list(folder.glob("*.png"))

    if not image_files:
        print(f"  ⚠ No images found in '{category_name}' — skipping")
        continue

    print(f"  → [{category_name}] Processing {len(image_files)} images...", end="", flush=True)

    vectors = []
    for img_path in image_files:
        vec = get_clip_vector(img_path)
        if vec is not None:
            vectors.append(vec)

    if not vectors:
        print(" ⚠ All images failed — skipping")
        continue

    # Average all vectors → Centroid
    centroid = np.mean(np.array(vectors), axis=0).tolist()

    food_img_collection.add(
        ids        = [f"centroid_{category_name}"],
        embeddings = [centroid],
        metadatas  = [{
            "category":   category_name,
            "num_images": len(vectors),
            "type":       "food_image_centroid"
        }],
        documents  = [f"Food category: {category_name.replace('_', ' ').title()}"]
    )
    centroids_added += 1
    print(f" ✓ Centroid stored ({len(vectors)} images averaged)")

print(f"\n  ✅ PHASE 1 DONE — {centroids_added}/{len(category_folders)} centroids stored in ChromaDB")


# ─────────────────────────────────────────────────────────
# PHASE 2 & 3 — HELPER: OpenAI Batch Embedder
# ─────────────────────────────────────────────────────────
openai_client = OpenAI(api_key=OPENAI_API_KEY)
EMBED_MODEL   = "text-embedding-3-small"

def embed_batch(texts: list[str]) -> list[list[float]]:
    """Call OpenAI embeddings API with retry on rate-limit."""
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = openai_client.embeddings.create(
                model = EMBED_MODEL,
                input = texts
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            wait = 2 ** attempt
            print(f"\n    ⚠ OpenAI error (attempt {attempt+1}): {e} — retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError("OpenAI embeddings failed after max retries.")


def build_and_store(
    collection_name: str,
    json_path: Path,
    text_builder,          # fn(item) → str
    id_key: str = "id",
    batch_size: int = 100
):
    """Generic function to read a JSON, embed text, and store in ChromaDB."""
    print(f"\n  → Loading {json_path.name}...")
    with open(json_path, "r", encoding="utf-8") as f:
        items = json.load(f)
    print(f"  → {len(items)} items loaded")

    # Clear old collection
    try:
        chroma_client.delete_collection(collection_name)
        print(f"  → Cleared existing '{collection_name}' collection")
    except Exception:
        pass

    collection = chroma_client.get_or_create_collection(
        name     = collection_name,
        metadata = {"hnsw:space": "cosine"}
    )

    total   = len(items)
    stored  = 0
    skipped = 0

    for start in range(0, total, batch_size):
        batch = items[start : start + batch_size]

        texts = []
        valid = []
        for item in batch:
            text = text_builder(item)
            if text and text.strip():
                texts.append(text)
                valid.append(item)
            else:
                skipped += 1

        if not texts:
            continue

        embeddings = embed_batch(texts)

        collection.add(
            ids        = [str(it[id_key]) for it in valid],
            embeddings = embeddings,
            metadatas  = [
                {k: (str(v) if isinstance(v, (list, dict)) else v)
                 for k, v in it.items()}
                for it in valid
            ],
            documents  = texts
        )
        stored += len(valid)

        if stored % 1000 == 0 or start == 0:
            pct = (start + len(batch)) / total * 100
            print(f"    Progress: {stored}/{total} stored ({pct:.1f}%)", flush=True)

    print(f"  ✅ Done — {stored} items stored, {skipped} skipped")
    return stored


# ─────────────────────────────────────────────────────────
# PHASE 2: FOOD TEXT EMBEDDINGS
# ─────────────────────────────────────────────────────────
print("\n[PHASE 2] Food Text Embeddings via OpenAI")
print("-" * 40)

def food_text_builder(item: dict) -> str:
    """Build a rich descriptive text string for a food item."""
    parts = []
    if name := item.get("food_name", ""):
        parts.append(f"Food: {name}")
    if cal := item.get("calories_kcal"):
        parts.append(f"Calories: {cal} kcal")
    if item.get("is_high_protein"):
        parts.append("High protein food")
    if src := item.get("data_source", ""):
        # Clean up source tag
        src = src.replace("_", " ").replace(".csv", "").replace(".json", "")
        parts.append(f"Source: {src}")
    return ". ".join(parts)

build_and_store(
    collection_name = "food_text",
    json_path       = FOOD_JSON,
    text_builder    = food_text_builder,
    id_key          = "id"
)


# ─────────────────────────────────────────────────────────
# PHASE 3: EXERCISE TEXT EMBEDDINGS
# ─────────────────────────────────────────────────────────
print("\n[PHASE 3] Exercise Text Embeddings via OpenAI")
print("-" * 40)

def exercise_text_builder(item: dict) -> str:
    """Build a rich descriptive text string for an exercise."""
    parts = []
    if name := item.get("name", ""):
        parts.append(f"Exercise: {name}")
    if muscle := item.get("main_muscle", ""):
        parts.append(f"Primary muscle: {muscle}")
    if equip := item.get("equipment", ""):
        parts.append(f"Equipment: {equip}")
    if targets := item.get("target_muscles", []):
        if isinstance(targets, list):
            parts.append(f"Target muscles: {', '.join(targets)}")
        else:
            parts.append(f"Target muscles: {targets}")
    if synergists := item.get("synergist_muscles", []):
        if isinstance(synergists, list):
            parts.append(f"Synergist muscles: {', '.join(synergists)}")
        else:
            parts.append(f"Synergist muscles: {synergists}")
    if prep := item.get("preparation", ""):
        parts.append(f"Preparation: {prep}")
    if exe := item.get("execution", ""):
        parts.append(f"Execution: {exe}")
    return ". ".join(parts)

build_and_store(
    collection_name = "exercise_text",
    json_path       = EXERCISE_JSON,
    text_builder    = exercise_text_builder,
    id_key          = "id"
)


# ─────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  ✅ ALL PHASES COMPLETE")
print("=" * 60)

collections = chroma_client.list_collections()
print("\n  ChromaDB Collections created:")
for col in collections:
    c = chroma_client.get_collection(col.name)
    print(f"    • {col.name:<30}  {c.count()} vectors")

print(f"\n  📦 Database saved to: {CHROMA_DIR}")
print("\n  The system is ready for querying!")
print("=" * 60)
