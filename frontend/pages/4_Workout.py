"""
Phase 3 - Standalone Workout Plan Generator.

Directly queries the Training Agent endpoint `/api/ai/generate-workout`
to generate structured workout cards and customized exercises.
Pre-populates fields using the cached profile to minimize manual typing.
"""
import streamlit as st

from lib import auth
from lib.api_client import ApiError, post
from lib.debug import is_debug
from lib.renderers import render_agent_response

st.set_page_config(
    page_title="Generate Workout - AI Fitness Gym",
    page_icon="🏋️",
    layout="wide",
)

# Enforce profile context first
auth.require_profile()

st.title("🏋️ Workout Plan Generator")
st.caption(
    "Direct access to the Training Agent to build a highly targeted workout program. "
    "Values are pre-filled from your profile context."
)

# Fetch user profile to pre-fill the form
profile = auth.get_profile_data() or {}

# Setup Form
with st.form("workout_generation_form", clear_on_submit=False):
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("🎯 Program Goals & Experience")
        
        # Mapping goal choices
        goal_choices = ["fat_loss", "muscle_gain", "maintenance", "athletic_performance", "Other"]
        stored_goal = profile.get("goal", "fat_loss")
        goal_index = 0
        if stored_goal in goal_choices:
            goal_index = goal_choices.index(stored_goal)
            
        goal = st.selectbox(
            "Primary Focus Goal",
            options=goal_choices,
            index=goal_index,
            help="Select the fitness goal to tailor exercise intensity and splits.",
        )
        
        level = st.selectbox(
            "Experience Level",
            options=["beginner", "intermediate", "advanced"],
            index=1,  # Default intermediate
            help="Determines the complexity and target volume of the workouts.",
        )
        
        # Accommodate Injuries
        injuries_list = profile.get("injuries") or []
        injuries_str = ", ".join(injuries_list) if isinstance(injuries_list, list) else str(injuries_list)
        injuries_input = st.text_input(
            "Injuries / Restrictions (Optional)",
            value=injuries_str,
            placeholder="e.g. lower back pain, left knee injury",
            help="The agent will actively substitute unsafe exercises to protect these joints.",
        )

    with col_b:
        st.subheader("📊 Your Biometrics")
        
        # Read profile biometrics safely
        weight = float(profile.get("weight") or 70.0)
        height = float(profile.get("height") or 170.0)
        age = int(profile.get("age") or 25)
        gender = str(profile.get("gender") or "male")
        
        st.markdown(f"**Age:** {age} years")
        st.markdown(f"**Gender:** {gender.title()}")
        st.markdown(f"**Height:** {height} cm")
        st.markdown(f"**Weight:** {weight} kg")
        st.caption("ℹ️ Update these values anytime on your Profile page.")
        
        st.divider()
        st.subheader("📝 Custom Instructions")
        custom_msg = st.text_area(
            "Coaching Directives & Special Requests",
            placeholder="e.g. 'I want a 3-day full body split using dumbbells only', 'Make it a home workout with no equipment', etc.",
            help="Give the AI coach specific guidance on equipment, duration, or focus areas.",
        )

    st.divider()
    submitted = st.form_submit_button("Generate Workout Program", type="primary", use_container_width=True)

# Handle Form Submission
if submitted:
    # Prepare list of injuries
    clean_injuries = [i.strip() for i in injuries_input.split(",") if i.strip()] if injuries_input else []
    
    payload = {
        "goal": goal,
        "level": level,
        "gender": gender,
        "age": age,
        "height": int(height),
        "weight": int(weight),
        "injuries": clean_injuries,
        "message": custom_msg.strip() if custom_msg else None,
    }
    
    with st.spinner("AI Training Specialist is drafting your program..."):
        try:
            data = post(
                "/api/ai/generate-workout",
                json=payload,
                timeout=180,
            )
            
            st.success("✅ Workout plan generated successfully!")
            
            # Render returned response cleanly using core system components
            render_agent_response(data)
            
            if is_debug():
                with st.expander("🔬 Raw API Response", expanded=False):
                    st.json(data)
                    
        except ApiError as e:
            st.error(f"Failed to generate workout plan: {e}")
