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
        
        # Two types of mappings + text instructions
        self.gifs: Dict[str, str] = {}
        self.images: Dict[str, str] = {}
        self.instructions: Dict[str, str] = {}
        
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
        # Depluralize: strip trailing 's' from each token (dynamic, universal)
        tokens = name.split()
        depluralized = []
        for t in tokens:
            if len(t) > 3 and t.endswith('s') and not t.endswith('ss'):
                depluralized.append(t[:-1])
            else:
                depluralized.append(t)
        return ' '.join(depluralized)

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
                steps = item.get("steps", [])
                
                if not name: continue
                norm_name = self._normalize(name)
                
                if steps and isinstance(steps, list):
                    self.instructions[norm_name] = " ".join(steps)
                    

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
        Returns a dict with 'gif', 'image', and 'instructions' paths/strings if found.
        """
        # LAZY LOAD on first call
        if not self.gifs and not self.images:
            self._load_mappings()

        norm_name = self._normalize(exercise_name)
        result = {"gif": None, "image": None, "instructions": None}
        
        result["gif"] = self.gifs.get(norm_name)
        result["image"] = self.images.get(norm_name)
        result["instructions"] = self.instructions.get(norm_name)

        if not result["gif"] or not result["image"] or not result["instructions"]:
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
                        
        # ── LOOSER FALLBACK (Level 3) ──
        # If still not found, check if a major part of the name (like 'bench press') matches
        if not result["gif"] or not result["image"]:
            tokens = norm_name.split()
            # Try to match the last 2 or 3 words (e.g., 'bench press' from 'board bench press')
            for i in range(len(tokens) - 1):
                sub_phrase = " ".join(tokens[i:])
                if len(sub_phrase) > 5:  # avoid matching small words
                    if not result["gif"]:
                        for key in self.gifs.keys():
                            if sub_phrase in key:
                                result["gif"] = self.gifs[key]
                                break
                    if not result["image"]:
                        for key in self.images.keys():
                            if sub_phrase in key:
                                result["image"] = self.images[key]
                                break
                if result["gif"] and result["image"]:
                    break

        # ── REVERSE SUBSTRING (Level 4) ──
        # Check if any DB key is contained WITHIN the normalized name (bidirectional)
        if not result["gif"] or not result["image"]:
            sorted_keys = sorted(self.gifs.keys(), key=len, reverse=True)
            for key in sorted_keys:
                if len(key) > 5 and key in norm_name:
                    if not result["gif"] and key in self.gifs:
                        result["gif"] = self.gifs[key]
                    if not result["image"] and key in self.images:
                        result["image"] = self.images[key]
                    if result["gif"] and result["image"]:
                        break

        # ── SINGLE-TOKEN FALLBACK (Level 5) ──
        # For short exercise names (e.g., 'deadlift'), check if norm_name exists inside any DB key
        if not result["gif"] or not result["image"]:
            if len(norm_name) > 4:
                # Sort by length (shortest first) to prefer closest match
                sorted_gif_keys = sorted(self.gifs.keys(), key=len)
                for key in sorted_gif_keys:
                    if norm_name in key:
                        if not result["gif"]:
                            result["gif"] = self.gifs[key]
                        if not result["image"] and key in self.images:
                            result["image"] = self.images[key]
                        break
                if not result["image"]:
                    sorted_img_keys = sorted(self.images.keys(), key=len)
                    for key in sorted_img_keys:
                        if norm_name in key:
                            result["image"] = self.images[key]
                            break

        # ── TOKEN OVERLAP (Level 6) ──
        # For cases like 'dumbbell row' matching 'dumbbell bent over row'
        # where all query tokens exist in the DB key but not contiguously
        if not result["gif"] or not result["image"]:
            query_tokens = set(norm_name.split())
            if len(query_tokens) >= 2:
                best_key = None
                best_score = 0
                for key in self.gifs.keys():
                    key_tokens = set(key.split())
                    overlap = query_tokens & key_tokens
                    if len(overlap) == len(query_tokens) and len(overlap) >= 2:
                        # All query tokens found in this key; prefer shortest key
                        score = len(overlap) / len(key_tokens)
                        if score > best_score:
                            best_score = score
                            best_key = key
                if best_key:
                    if not result["gif"]:
                        result["gif"] = self.gifs[best_key]
                    if not result["image"] and best_key in self.images:
                        result["image"] = self.images[best_key]

        if not result["instructions"] and result["gif"]:
            # Attempt to pull instruction from the matched fuzzy key
            for k, v in self.gifs.items():
                if v == result["gif"]:
                    result["instructions"] = self.instructions.get(k)
                    if result["instructions"]:
                        break

        return result

# Singleton instance
media_matcher = MediaMatcher()
