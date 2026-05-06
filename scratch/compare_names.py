import json
import random
from pathlib import Path

data_dir = Path("/Users/shubhamjadhav/Documents/Fit Bot/ai-fitness-app/Data")
master_path = data_dir / "final_master_exercises.json"
exercises_path = data_dir / "exercises-dataset" / "data" / "exercises (1).json"

with open(master_path, "r") as f:
    master = json.load(f)

with open(exercises_path, "r") as f:
    exercises = json.load(f)

print("--- Master Names Sample ---")
for item in random.sample(master, min(20, len(master))):
    print(item['name'])

print("\n--- GIF Dataset Names Sample ---")
for item in random.sample(exercises, min(20, len(exercises))):
    print(item['name'])
