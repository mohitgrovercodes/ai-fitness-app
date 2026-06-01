import asyncio
import json
import logging
from pathlib import Path
from typing import List

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import ValidationError

from app.core.config import settings
from app.safety.refusal import SegmentedTags

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")

MASTER_DB_PATH = Path("d:/AI/IMGProjects/ai-fitness-app/ai-fitness-app/data/final_master_exercises.json")
OUTPUT_DB_PATH = Path("d:/AI/IMGProjects/ai-fitness-app/ai-fitness-app/app/safety/tags_master.json")

PROMPT = """You are an expert sports biomechanist. Your job is to classify fitness exercises into a rigid 10-feature safety constraint model.

Given the exercise details below, provide the SegmentedTags.
Exercise: {name}
Preparation: {prep}
Execution: {exec}
Target Muscles: {target}
Synergists: {synergist}
Main Muscle: {main}

Instructions for Fields:
- primary_joints_involved: Only joints that actively flex/extend/rotate under load as prime movers.
- kinetic_chain_loading: CLOSED_LOADED (e.g. squat, pushup), CLOSED_SUPPORTED (e.g. leg press, machine row), OPEN_UNLOADED (e.g. leg extension, bicep curl).
- axial_compression_level: 0=NONE (lying down/seated back support), 1=MEDIUM (seated no support, standing light), 2=HIGH (heavy standing, squat, overhead press).
- grip_requirement: 0=NONE, 1=LIGHT, 2=HEAVY.
- joint_impact_level: 0=NONE, 1=LOW, 2=HIGH (plyometrics/jumping).
- upper_limb_stabilization: ACTIVE (bearing bodyweight/plank/pushup) or NONE.
- metabolic_density: 0=LOW (isolation), 1=MEDIUM, 2=HIGH (compound/full body).
- torsional_joint_loading: true if there is deliberate rotation/twisting/pivoting of joints (e.g. Russian twists, rotational lunges).
- spinal_shear_level: 0=NONE, 1=MEDIUM (bent over row), 2=HIGH (deadlift, RDL, good morning).
- joint_actions: Explicit joint motions (e.g. KNEE_EXTENSION_OPEN_CHAIN for leg extension, KNEE_EXTENSION_CLOSED_CHAIN for squat).
- primary_segment: The single dominant muscle group (LOWER_ANTERIOR, LOWER_POSTERIOR, LOWER_LATERAL, CORE_ANTERIOR, CORE_POSTERIOR, UPPER_PUSH, UPPER_PULL).
- secondary_segments: Additional segments significantly activated.
"""

async def tag_exercise(llm, ex_data: dict, semaphore: asyncio.Semaphore) -> dict | None:
    async with semaphore:
        try:
            chain = ChatPromptTemplate.from_template(PROMPT) | llm
            result: SegmentedTags = await chain.ainvoke({
                "name": ex_data.get("name", ""),
                "prep": ex_data.get("preparation", ""),
                "exec": ex_data.get("execution", ""),
                "target": ", ".join(ex_data.get("target_muscles", [])),
                "synergist": ", ".join(ex_data.get("synergist_muscles", [])),
                "main": ex_data.get("main_muscle", "")
            })
            
            # Ensure exercise_id and name are exactly as in original data
            data_dump = result.model_dump()
            data_dump["exercise_id"] = ex_data["id"]
            data_dump["name"] = ex_data["name"]
            
            logger.info(f"✅ Tagged: {ex_data['name']} ({ex_data['id']}) -> {data_dump['primary_segment']}")
            return data_dump
        except Exception as e:
            logger.error(f"❌ Failed to tag {ex_data['name']} ({ex_data['id']}): {e}")
            return None

async def run_auto_tagger(batch_size: int = 50, limit: int = None):
    with open(MASTER_DB_PATH, "r", encoding="utf-8") as f:
        master_exercises = json.load(f)
        
    existing_tags = {}
    if OUTPUT_DB_PATH.exists():
        with open(OUTPUT_DB_PATH, "r", encoding="utf-8") as f:
            for item in json.load(f):
                existing_tags[item["exercise_id"]] = item
                
    exercises_to_tag = [ex for ex in master_exercises if ex["id"] not in existing_tags]
    if limit:
        exercises_to_tag = exercises_to_tag[:limit]
        
    logger.info(f"Total exercises: {len(master_exercises)}. Already tagged: {len(existing_tags)}. Remaining to tag in this run: {len(exercises_to_tag)}")
    
    if not exercises_to_tag:
        logger.info("Nothing to tag!")
        return

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.0,
        api_key=settings.OPENAI_API_KEY,
        max_retries=3,
    ).with_structured_output(SegmentedTags, method="function_calling")
    
    semaphore = asyncio.Semaphore(15) # Concurrent API calls
    
    # Process in batches to save state incrementally
    for i in range(0, len(exercises_to_tag), batch_size):
        batch = exercises_to_tag[i:i+batch_size]
        logger.info(f"Processing batch {i} to {i+len(batch)}...")
        
        tasks = [tag_exercise(llm, ex, semaphore) for ex in batch]
        results = await asyncio.gather(*tasks)
        
        # Save successes
        for res in results:
            if res:
                existing_tags[res["exercise_id"]] = res
                
        # Write to disk incrementally
        with open(OUTPUT_DB_PATH, "w", encoding="utf-8") as f:
            json.dump(list(existing_tags.values()), f, indent=2, ensure_ascii=False)
            
        logger.info(f"💾 Saved batch. Total tagged: {len(existing_tags)}")
        
if __name__ == "__main__":
    # For testing, we can limit it to 20. If limit=None, it does all of them.
    # We will do a full run.
    asyncio.run(run_auto_tagger())
