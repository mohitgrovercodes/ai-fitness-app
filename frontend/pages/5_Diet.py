"""
Phase 3 - Standalone Diet Plan Generator.

Directly queries the Nutrition Agent endpoint `/api/ai/generate-diet`
to generate structured meal plan cards and dynamic daily totals.
Pre-populates fields using the cached profile to minimize manual typing.
"""
import streamlit as st

from lib import auth
from lib.api_client import ApiError, post
from lib.debug import is_debug
from lib.renderers import render_agent_response

st.set_page_config(
    page_title="Generate Diet - AI Fitness Gym",
    page_icon="🥗",
    layout="wide",
)

# Enforce profile context first
auth.require_profile()

st.title("🥗 Diet Plan Generator")
st.caption(
    "Direct access to the Nutrition Agent to design a customized meal plan. "
    "Values are pre-filled from your profile context."
)

# Fetch user profile to pre-fill the form
profile = auth.get_profile_data() or {}

# Setup Form
with st.form("diet_generation_form", clear_on_submit=False):
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("🎯 Diet Preferences & Exclusions")
        
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
            help="Select the goal to compute proper calorie deficit/surplus macros.",
        )
        
        # Dietary choices matching profile enums
        diet_choices = ["vegetarian", "non_vegetarian", "vegan", "jain", "eggetarian", "keto", "paleo"]
        stored_diet = profile.get("diet_preference", "vegetarian")
        diet_index = 0
        if stored_diet in diet_choices:
            diet_index = diet_choices.index(stored_diet)
            
        diet_type = st.selectbox(
            "Dietary Profile",
            options=diet_choices,
            index=diet_index,
            help="Your primary cooking / consumption standard.",
        )
        
        # Allergies
        allergies_list = profile.get("allergies") or []
        allergies_str = ", ".join(allergies_list) if isinstance(allergies_list, list) else str(allergies_list)
        allergies_input = st.text_input(
            "Food Allergies to Exclude (Optional)",
            value=allergies_str,
            placeholder="e.g. peanuts, dairy, shellfish",
            help="The agent will exclude these ingredients completely from your recipes.",
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
            "Meal Directives & Preference Details",
            placeholder="e.g. 'Generate a high-protein diet with a lot of oats and cottage cheese', 'Make a simple 3-meal plan with low prep times', etc.",
            help="Specify details like number of meals, snack preferences, or specific ingredients.",
        )

    st.divider()
    submitted = st.form_submit_button("Generate Diet Program", type="primary", use_container_width=True)

# Handle Form Submission
if submitted:
    # Prepare list of allergies
    clean_allergies = [a.strip() for a in allergies_input.split(",") if a.strip()] if allergies_input else []
    
    payload = {
        "goal": goal,
        "diet_type": diet_type,
        "gender": gender,
        "age": age,
        "height": int(height),
        "weight": int(weight),
        "allergies": clean_allergies,
        "message": custom_msg.strip() if custom_msg else None,
    }
    
    with st.spinner("AI Nutrition Specialist is calculating your targets and meals..."):
        try:
            data = post(
                "/api/ai/generate-diet",
                json=payload,
                timeout=180,
            )
            
            st.success("✅ Diet plan generated successfully!")
            
            # Render returned response cleanly using core system components
            render_agent_response(data)
            
            if is_debug():
                with st.expander("🔬 Raw API Response", expanded=False):
                    st.json(data)
                    
        except ApiError as e:
            st.error(f"Failed to generate diet plan: {e}")
