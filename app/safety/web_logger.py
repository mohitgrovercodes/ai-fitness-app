import asyncio
import json
import logging
from typing import List, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app.core.config import settings
from app.safety.refusal import SegmentedTags
from app.safety.auto_tagger import OUTPUT_DB_PATH

logger = logging.getLogger(__name__)

class WebExtractedExercise(BaseModel):
    name: str = Field(description="Name of the exercise found in the web text")
    description: str = Field(description="Brief execution description from the web text")

class WebExtractionList(BaseModel):
    exercises: List[WebExtractedExercise] = Field(description="List of exercises extracted from the web context")

async def extract_exercises_from_web(query: str, web_context: str) -> List[WebExtractedExercise]:
    """Parses raw web context to find new exercises mentioned."""
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.0,
        api_key=settings.OPENAI_API_KEY
    ).with_structured_output(WebExtractionList, method="function_calling")
    
    prompt = f"""You are analyzing a web search result for fitness exercises.
    User Query: {query}
    Web Context:
    {web_context}
    
    Extract a list of all distinct physical fitness exercises mentioned in the context.
    Do not include generic terms like 'workout' or 'cardio'. Be specific (e.g., 'Zercher Squat', 'Bulgarian Split Squat').
    """
    
    try:
        result = await llm.ainvoke(prompt)
        return result.exercises
    except Exception as e:
        logger.error(f"❌ [WebLogger] Extraction failed: {e}")
        return []

async def tag_web_exercise(ex: WebExtractedExercise) -> Optional[dict]:
    """Uses the auto-tagger logic to generate SegmentedTags for the new exercise."""
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.0,
        api_key=settings.OPENAI_API_KEY
    ).with_structured_output(SegmentedTags, method="function_calling")
    
    from app.safety.auto_tagger import PROMPT
    
    chain = ChatPromptTemplate.from_template(PROMPT) | llm
    
    try:
        result: SegmentedTags = await chain.ainvoke({
            "name": ex.name,
            "prep": "Unknown (derived from web)",
            "exec": ex.description,
            "target": "Unknown",
            "synergist": "Unknown",
            "main": "Unknown"
        })
        
        data_dump = result.model_dump()
        # Generate a unique ID for web-found exercises
        safe_name = ex.name.lower().replace(" ", "_").replace("-", "_")
        data_dump["exercise_id"] = f"web_{safe_name}"
        data_dump["name"] = ex.name
        
        return data_dump
    except Exception as e:
        logger.error(f"❌ [WebLogger] Tagging failed for {ex.name}: {e}")
        return None

async def run_web_extraction_pipeline(query: str, web_context: str):
    """Background pipeline: extracts, tags, and saves new exercises."""
    logger.info("🌐 [WebLogger] Starting background extraction pipeline...")
    extracted = await extract_exercises_from_web(query, web_context)
    
    if not extracted:
        logger.info("🌐 [WebLogger] No exercises found in web context.")
        return

    logger.info(f"🌐 [WebLogger] Extracted {len(extracted)} exercises: {[e.name for e in extracted]}")
    
    # Load existing tags to avoid duplicates
    existing_ids = set()
    existing_names = set()
    if OUTPUT_DB_PATH.exists():
        try:
            with open(OUTPUT_DB_PATH, "r", encoding="utf-8") as f:
                current_data = json.load(f)
                for item in current_data:
                    existing_ids.add(item["exercise_id"])
                    existing_names.add(item["name"].lower())
        except Exception:
            pass

    new_tags = []
    for ex in extracted:
        safe_name = ex.name.lower().replace(" ", "_").replace("-", "_")
        ex_id = f"web_{safe_name}"
        if ex_id in existing_ids or ex.name.lower() in existing_names:
            logger.info(f"🌐 [WebLogger] '{ex.name}' already exists in DB. Skipping.")
            continue
            
        logger.info(f"🌐 [WebLogger] Tagging new web exercise: {ex.name}...")
        tagged = await tag_web_exercise(ex)
        if tagged:
            new_tags.append(tagged)
            
    if new_tags:
        # Append to tags_master.json safely
        try:
            current_data = []
            if OUTPUT_DB_PATH.exists():
                with open(OUTPUT_DB_PATH, "r", encoding="utf-8") as f:
                    current_data = json.load(f)
            
            current_data.extend(new_tags)
            
            with open(OUTPUT_DB_PATH, "w", encoding="utf-8") as f:
                json.dump(current_data, f, indent=2, ensure_ascii=False)
                
            logger.info(f"✅ [WebLogger] Added {len(new_tags)} new exercises to {OUTPUT_DB_PATH.name}!")
        except Exception as e:
            logger.error(f"❌ [WebLogger] Failed to write to {OUTPUT_DB_PATH.name}: {e}")
