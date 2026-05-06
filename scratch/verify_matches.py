import json
import re
from pathlib import Path

data_dir = Path("/Users/shubhamjadhav/Documents/Fit Bot/ai-fitness-app/Data")
master_path = data_dir / "final_master_exercises.json"
mapping_path = data_dir / "exercise_gif_mapping.json"

with open(master_path, "r") as f:
    master = json.load(f)

with open(mapping_path, "r") as f:
    mapping = json.load(f)

def normalize(name):
    name = name.lower()
    name = re.sub(r'[^a-z0-9]', ' ', name)
    name = ' '.join(name.split())
    return name

matches = 0
for ex in master:
    norm_name = normalize(ex['name'])
    if norm_name in mapping:
        matches += 1

print(f"Master items: {len(master)}")
print(f"Matches in mapping: {matches}")
