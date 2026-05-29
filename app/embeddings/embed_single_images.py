import os
import glob
from pathlib import Path

import chromadb
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import warnings

warnings.filterwarnings("ignore")

# Paths
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
CHROMA_DIR = ROOT_DIR / "chromadb_store"
SINGLE_IMAGES_DIR = ROOT_DIR / "Data/Food-dataset/food/data/data"

# Connect to DB
chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
image_col = chroma_client.get_collection("food_image_centroids")

# Load CLIP
print("⏳ Loading AI Vision model...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()

# Get all images
image_paths = glob.glob(str(SINGLE_IMAGES_DIR / "*.[jJ][pP][gG]"))
image_paths += glob.glob(str(SINGLE_IMAGES_DIR / "*.[jJ][pP][eE][gG]"))
image_paths += glob.glob(str(SINGLE_IMAGES_DIR / "*.[pP][nN][gG]"))

print(f"\nFound {len(image_paths)} single images to embed.")

BATCH_SIZE = 100
stored = 0

print("🚀 Starting embedding process...")

for i in range(0, len(image_paths), BATCH_SIZE):
    batch_paths = image_paths[i:i + BATCH_SIZE]
    ids = []
    embeddings = []
    metadatas = []
    
    for path in batch_paths:
        filename = os.path.basename(path)
        img_id = f"single_{filename}"
        
        try:
            img = Image.open(path).convert("RGB")
            inputs = clip_processor(images=img, return_tensors="pt", padding=True)
            with torch.no_grad():
                vec = clip_model.get_image_features(**inputs)
            
            # Normalize
            vec = vec / vec.norm(dim=-1, keepdim=True)
            vector = vec.squeeze().numpy().tolist()
            
            # Create a clean readable name from the filename for text searching later
            clean_name = filename.replace("_", " ").replace("-", " ").replace(".jpg", "").replace(".jpeg", "").replace(".png", "")
            # Remove leading numbers (e.g., "1. Doddapatre..." -> "Doddapatre...")
            if clean_name.split(".")[0].isdigit():
                clean_name = " ".join(clean_name.split(".")[1:]).strip()
            
            ids.append(img_id)
            embeddings.append(vector)
            metadatas.append({
                "type": "single_image",
                "category": clean_name, # We store the cleaned name as the category
                "original_filename": filename
            })
            
        except Exception as e:
            print(f"  ❌ Skipped {filename}: {e}")
            continue
            
    if ids:
        image_col.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas
        )
        stored += len(ids)
        print(f"  → Embedded {stored}/{len(image_paths)} images")

print(f"\n✅ DONE! Successfully embedded and stored {stored} single images.")
