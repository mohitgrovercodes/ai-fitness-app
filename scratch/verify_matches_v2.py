import json
import re
from pathlib import Path

data_dir = Path("/Users/shubhamjadhav/Documents/Fit Bot/ai-fitness-app/Data")
master_path = data_dir / "final_master_exercises.json"
exercises_path = data_dir / "exercises-dataset" / "data" / "exercises (1).json"

with open(master_path, "r") as f:
    master = json.load(f)

with open(exercises_path, "r") as f:
    exercises = json.load(f)

def normalize(name):
    name = name.lower()
    # Remove known prefixes
    prefixes = ["metaburn", "holman", "rusin", "30 arms", "30 shoulders", "dumbbell fix", "fyr", "boss everline", "hm", "un", "fyr2"]
    for p in prefixes:
        if name.startswith(p):
            name = name[len(p):].strip()
    
    name = re.sub(r'[^a-z0-9]', ' ', name)
    name = ' '.join(name.split())
    return name

mapping = {}
for ex in exercises:
    norm_name = normalize(ex['name'])
    mapping[norm_name] = ex['gif']

matches = 0
matched_samples = []

for ex in master:
    m_name = normalize(ex['name'])
    
    # 1. Exact match
    if m_name in mapping:
        matches += 1
        matched_samples.append((ex['name'], m_name, mapping[m_name]))
        continue
        
    # 2. Contains match (Check if any GIF name is inside the Master name)
    found = False
    for g_name, g_path in mapping.items():
        if len(g_name) > 5 and g_name in m_name:
            matches += 1
            matched_samples.append((ex['name'], g_name, g_path))
            found = True
            break
    if found:
        continue

print(f"Master items: {len(master)}")
print(f"Matches in mapping: {matches}")
print("\n--- Match Samples ---")
for original, normalized, gif in matched_samples[:20]:
    print(f"Original: {original} | Match: {normalized} | GIF: {gif}")
