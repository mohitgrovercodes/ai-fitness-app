import os
import json
from pathlib import Path
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import chromadb
import warnings

warnings.filterwarnings("ignore")

# Paths
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
CHROMA_DIR = ROOT_DIR / "chromadb_store"
ARCHIVE_DIR = ROOT_DIR / "Data/Food-dataset/food/archive (7)/Indian Food Images/Indian Food Images"

# Connect to DB
chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
image_col = chroma_client.get_collection("food_image_centroids")

# Load CLIP
print("⏳ Loading CLIP model for 80 archive categories...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()

# Get the 80 categories
categories = [d for d in os.listdir(ARCHIVE_DIR) if os.path.isdir(ARCHIVE_DIR / d)]
categories.sort()

print(f"\nProcessing centroids for {len(categories)} archive categories...")

for cat in categories:
    cat_path = ARCHIVE_DIR / cat
    img_files = [f for f in os.listdir(cat_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not img_files:
        print(f"  ⚠ No images found for {cat}, skipping.")
        continue
        
    print(f"  → [{cat}] Processing {len(img_files)} images...")
    
    cat_embeddings = []
    for img_name in img_files:
        try:
            img_path = cat_path / img_name
            img = Image.open(img_path).convert("RGB")
            inputs = clip_processor(images=img, return_tensors="pt", padding=True)
            with torch.no_grad():
                features = clip_model.get_image_features(**inputs)
            # Normalize
            features = features / features.norm(dim=-1, keepdim=True)
            cat_embeddings.append(features.squeeze().numpy())
        except Exception as e:
            continue
            
    if cat_embeddings:
        # Calculate Centroid
        centroid = torch.tensor(cat_embeddings).mean(dim=0)
        # Normalize Centroid
        centroid = centroid / centroid.norm(dim=-1, keepdim=True)
        centroid_vector = centroid.tolist()
        
        # Store in ChromaDB
        image_col.add(
            ids=[f"centroid_{cat}"],
            embeddings=[centroid_vector],
            metadatas=[{
                "type": "centroid",
                "category": cat,
                "source": "archive_7"
            }]
        )
        print(f"  ✓ Centroid stored for {cat}")

print("\n✅ ALL 80 ARCHIVE CATEGORIES EMBEDDED!")
