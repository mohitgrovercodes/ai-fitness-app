import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.database import db_manager

def delete_bad_thali():
    print("Connecting to DB to delete ONLY the hallucinated Thali...")
    client = db_manager._get_client()
    
    img_col = client.get_collection("food_image_centroids")
    text_col = client.get_collection("food_text")
    
    # 1. Delete from image collection
    try:
        results = img_col.get()
        ids_to_delete = [id for id in results["ids"] if "rajasthani_thali" in id]
        if ids_to_delete:
            img_col.delete(ids=ids_to_delete)
            print(f"✅ Deleted image vectors: {ids_to_delete}")
    except Exception as e:
        pass

    # 2. Delete from text collection
    try:
        results = text_col.get()
        ids_to_delete = [id for id in results["ids"] if "rajasthani_thali" in id]
        if ids_to_delete:
            text_col.delete(ids=ids_to_delete)
            print(f"✅ Deleted text vectors: {ids_to_delete}")
    except Exception as e:
        pass

if __name__ == "__main__":
    delete_bad_thali()
