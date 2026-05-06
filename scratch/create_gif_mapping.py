import json
import re
from pathlib import Path

data_dir = Path("/Users/shubhamjadhav/Documents/Fit Bot/ai-fitness-app/Data")
exercises_path = data_dir / "exercises-dataset" / "data" / "exercises (1).json"
output_path = data_dir / "exercise_gif_mapping.json"

with open(exercises_path, "r") as f:
    exercises = json.load(f)

def normalize(name):
    # Remove special characters, extra spaces, and convert to lower
    name = name.lower()
    name = re.sub(r'[^a-z0-9]', ' ', name)
    name = ' '.join(name.split())
    return name

mapping = {}
for ex in exercises:
    norm_name = normalize(ex['name'])
    mapping[norm_name] = ex['gif']

with open(output_path, "w") as f:
    json.dump(mapping, f, indent=2)

print(f"✅ Mapping created with {len(mapping)} items.")
