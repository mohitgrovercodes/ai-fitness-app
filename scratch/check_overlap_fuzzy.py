import json
from pathlib import Path

data_dir = Path("/Users/shubhamjadhav/Documents/Fit Bot/ai-fitness-app/Data")
master_path = data_dir / "final_master_exercises.json"
exercises_path = data_dir / "exercises-dataset" / "data" / "exercises (1).json"

with open(master_path, "r") as f:
    master = json.load(f)

with open(exercises_path, "r") as f:
    exercises = json.load(f)

master_names = {item['name'].lower().strip() for item in master}
exercises_names = {item['name'].lower().strip() for item in exercises}

overlap = master_names & exercises_names
print(f"Overlap: {len(overlap)}")
if overlap:
    print(f"Sample: {list(overlap)[:5]}")
else:
    # Try fuzzy match or check if one is a subset of the other
    print("Checking partial matches...")
    for m_name in list(master_names)[:100]:
        for e_name in list(exercises_names):
            if m_name in e_name or e_name in m_name:
                print(f"Match: '{m_name}' <-> '{e_name}'")
                break
