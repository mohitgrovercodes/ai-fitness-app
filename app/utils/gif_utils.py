import json
import re
from pathlib import Path
from typing import Optional, Dict
from app.utils.logger import logger

class GIFMatcher:
    """
    Matches exercise names to their corresponding GIF paths using normalized 
    exact matching and substring matching.
    """
    def __init__(self):
        self.root_dir = Path(__file__).resolve().parent.parent.parent
        self.mapping_file = self.root_dir / "Data" / "exercises-dataset" / "data" / "exercises (1).json"
        self.mapping: Dict[str, str] = {}
        self.prefixes = [
            "metaburn", "holman", "rusin", "30 arms", "30 shoulders", 
            "dumbbell fix", "fyr", "boss everline", "hm", "un", "fyr2"
        ]
        self._load_mapping()

    def _normalize(self, name: str) -> str:
        if not name:
            return ""
        name = name.lower()
        # Remove known prefixes
        for p in self.prefixes:
            if name.startswith(p):
                name = name[len(p):].strip()
        
        # Remove special characters and normalize whitespace
        name = re.sub(r'[^a-z0-9]', ' ', name)
        name = ' '.join(name.split())
        return name

    def _load_mapping(self):
        try:
            if not self.mapping_file.exists():
                logger.warning(f"⚠️ [GIF Matcher] Mapping file not found: {self.mapping_file}")
                return

            with open(self.mapping_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            videos_dir = self.root_dir / "Data" / "exercises-dataset"
            
            valid_count = 0
            for item in data:
                name = item.get("name")
                gif = item.get("gif")
                if name and gif:
                    # Loophole Fix: Verify file existence on disk
                    full_path = videos_dir / gif
                    if full_path.exists():
                        norm_name = self._normalize(name)
                        self.mapping[norm_name] = gif
                        valid_count += 1
            
            logger.info(f"✅ [GIF Matcher] Loaded {valid_count} verified GIF mappings.")
        except Exception as e:
            logger.error(f"❌ [GIF Matcher] Error loading mapping: {e}")

    def get_gif_path(self, exercise_name: str) -> Optional[str]:
        """
        Find a GIF path for a given exercise name.
        1. Try exact normalized match.
        2. Try whole-word substring match (e.g. 'Push up' matches 'MetaBurn Push up').
        """
        if not self.mapping:
            return None

        norm_name = self._normalize(exercise_name)
        
        # 1. Exact match
        if norm_name in self.mapping:
            return self.mapping[norm_name]
            
        # 2. Whole-word substring match
        # Sort by length descending to match the most specific name first
        sorted_keys = sorted(self.mapping.keys(), key=len, reverse=True)
        for key in sorted_keys:
            # Loophole Fix: Use regex for whole-word matching instead of simple 'in'
            # This avoids partial matches like 'run' matching 'crunch'
            # We also allow shorter names now (e.g. 'run', 'dip')
            pattern = rf"\b{re.escape(key)}\b"
            if re.search(pattern, norm_name):
                return self.mapping[key]
                
        return None

# Singleton instance
gif_matcher = GIFMatcher()
