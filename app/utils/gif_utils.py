import json
import re
from pathlib import Path
from typing import Optional, Dict
from app.utils.logger import logger

class MediaMatcher:
    """
    Matches exercise names to their corresponding GIF and Image paths.
    Verified against disk to ensure no broken links.
    """
    def __init__(self):
        self.root_dir = Path(__file__).resolve().parent.parent.parent
        self.dataset_dir = self.root_dir / "Data" / "exercises-dataset"
        self.mapping_file = self.dataset_dir / "data" / "exercises (1).json"
        
        # Two types of mappings
        self.gifs: Dict[str, str] = {}
        self.images: Dict[str, str] = {}
        
        self.prefixes = [
            "metaburn", "holman", "rusin", "30 arms", "30 shoulders", 
            "dumbbell fix", "fyr", "boss everline", "hm", "un", "fyr2"
        ]
        # Deferred load on first use in get_media()

    def _normalize(self, name: str) -> str:
        if not name:
            return ""
        name = name.lower()
        for p in self.prefixes:
            if name.startswith(p):
                name = name[len(p):].strip()
        name = re.sub(r'[^a-z0-9]', ' ', name)
        return ' '.join(name.split())

    def _load_mappings(self):
        try:
            if not self.mapping_file.exists():
                logger.warning(f"⚠️ [Media Matcher] Mapping file not found: {self.mapping_file}")
                return

            with open(self.mapping_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            for item in data:
                name = item.get("name")
                gif_rel = item.get("gif")
                img_rel = item.get("image")
                
                if not name: continue
                norm_name = self._normalize(name)

                # Verify and map GIF
                if gif_rel and (self.dataset_dir / gif_rel).exists():
                    self.gifs[norm_name] = gif_rel
                
                # Verify and map Image
                if img_rel and (self.dataset_dir / img_rel).exists():
                    self.images[norm_name] = img_rel
            
            logger.info(f"✅ [Media Matcher] Loaded {len(self.gifs)} GIFs and {len(self.images)} Images.")
        except Exception as e:
            logger.error(f"❌ [Media Matcher] Error: {e}")

    def get_media(self, exercise_name: str) -> Dict[str, Optional[str]]:
        """
        Returns a dict with 'gif' and 'image' paths if found.
        """
        # LAZY LOAD on first call
        if not self.gifs and not self.images:
            self._load_mappings()

        norm_name = self._normalize(exercise_name)
        result = {"gif": None, "image": None}
        # ... (rest of method remains same but uses self._load_mappings checks)
        result["gif"] = self.gifs.get(norm_name)
        result["image"] = self.images.get(norm_name)

        if not result["gif"] or not result["image"]:
            if not result["gif"]:
                sorted_gif_keys = sorted(self.gifs.keys(), key=len, reverse=True)
                for key in sorted_gif_keys:
                    if re.search(rf"\b{re.escape(key)}\b", norm_name):
                        result["gif"] = self.gifs[key]
                        break
            if not result["image"]:
                sorted_img_keys = sorted(self.images.keys(), key=len, reverse=True)
                for key in sorted_img_keys:
                    if re.search(rf"\b{re.escape(key)}\b", norm_name):
                        result["image"] = self.images[key]
                        break

        return result

# Singleton instance
media_matcher = MediaMatcher()
