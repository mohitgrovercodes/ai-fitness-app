import json
from pathlib import Path

data_dir = Path("/Users/shubhamjadhav/Documents/Fit Bot/ai-fitness-app/Data")
master_path = data_dir / "final_master_exercises.json"
exercises_path = data_dir / "exercises-dataset" / "data" / "exercises (1).json"

with open(master_path, "r") as f:
    master = json.load(f)

with open(exercises_path, "r") as f:
    exercises = json.load(f)

master_names = {item['name'].lower(): item for item in master}
exercises_names = {item['name'].lower(): item for item in exercises}

overlap = set(master_names.keys()) & set(exercises_names.keys())
print(f"Master items: {len(master)}")
print(f"Exercises items: {len(exercises)}")
print(f"Overlap by name: {len(overlap)}")

# Sample overlap
for name in list(overlap)[:5]:
    print(f"Name: {name}")
    print(f"  Master ID: {master_names[name].get('id')}")
    print(f"  Exercises ID: {exercises_names[name].get('id')}")
    print(f"  GIF: {exercises_names[name].get('gif')}")
