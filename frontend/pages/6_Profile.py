"""
Phase 1 — Profile page.

Behavior:
- On load, GET /api/profile/me.
  - 404 → show "Create your profile" form (onboarding mode).
  - 200 → show "Edit your profile" form, pre-filled with current values.
- Submit hits POST /api/profile/onboarding (create) or PATCH /api/profile/me (update).
- Goal and diet preference are free-form strings on the backend (normalized by
  Pydantic validators), so we expose curated dropdowns + an "Other (type your own)"
  escape hatch.
"""
import streamlit as st

from lib import auth
from lib.api_client import ApiError, get, post, patch
from lib.debug import render_response_panel

st.set_page_config(page_title="Profile · AI Fitness Gym", page_icon="👤", layout="wide")
auth.require_auth()

# ── Choices that match app/modules/profile/schema.py enums + mappings ────────
GENDER_CHOICES = ["male", "female", "other"]
ACTIVITY_CHOICES = [
    "sedentary",
    "lightly_active",
    "moderately_active",
    "very_active",
    "extra_active",
]
GOAL_CHOICES = [
    "fat_loss",
    "muscle_gain",
    "maintenance",
    "athletic_performance",
    "Other (type your own)",
]
DIET_CHOICES = [
    "vegetarian",
    "non_vegetarian",
    "vegan",
    "jain",
    "eggetarian",
    "keto",
    "paleo",
    "Other (type your own)",
]

OTHER_SENTINEL = "Other (type your own)"


def _csv_to_list(text: str) -> list[str]:
    """Comma-separated text → cleaned list. Empty input → []."""
    if not text:
        return []
    return [item.strip() for item in text.split(",") if item.strip()]


def _list_to_csv(items) -> str:
    if not items:
        return ""
    return ", ".join(str(i) for i in items)


def _safe_index(options: list[str], value, default: int = 0) -> int:
    """Return the index of `value` in `options`, or `default` if not present."""
    if value in options:
        return options.index(value)
    return default


def _load_existing_profile() -> dict | None:
    """Return the current user's profile dict, or None if not yet created."""
    try:
        return get("/api/profile/me")
    except ApiError as e:
        if e.status_code == 404:
            return None
        raise


def render_profile_form(existing: dict | None) -> None:
    """
    Single form used for both onboarding (existing is None) and edit
    (existing is the profile dict returned by /api/profile/me).
    """
    is_edit = existing is not None
    submit_label = "Save changes" if is_edit else "Create profile"
    heading = "👤 Edit your profile" if is_edit else "👤 Create your profile"

    st.title(heading)
    if is_edit:
        st.caption("Update any field below — only changed values are sent to the API.")
    else:
        st.caption(
            "Welcome! Tell us about yourself so the agents can personalize your plans."
        )

    existing = existing or {}

    # Pre-fill goal / diet selectboxes — fall back to "Other" if the stored
    # value isn't in our curated list (e.g. user typed "shredded" earlier).
    goal_stored = existing.get("goal", "")
    goal_index = _safe_index(GOAL_CHOICES, goal_stored, default=len(GOAL_CHOICES) - 1)

    diet_stored = existing.get("diet_preference", "")
    diet_index = _safe_index(DIET_CHOICES, diet_stored, default=len(DIET_CHOICES) - 1)

    with st.form("profile_form", clear_on_submit=False):
        col_a, col_b = st.columns(2)

        with col_a:
            full_name = st.text_input(
                "Full name *",
                value=existing.get("full_name", ""),
                max_chars=100,
            )
            age = st.number_input(
                "Age *",
                min_value=1,
                max_value=120,
                value=int(existing.get("age") or 25),
                step=1,
            )
            gender = st.selectbox(
                "Gender *",
                GENDER_CHOICES,
                index=_safe_index(GENDER_CHOICES, existing.get("gender", "male")),
            )
            height = st.number_input(
                "Height (cm) *",
                min_value=50.0,
                max_value=250.0,
                value=float(existing.get("height") or 170.0),
                step=0.5,
            )
            weight = st.number_input(
                "Weight (kg) *",
                min_value=10.0,
                max_value=300.0,
                value=float(existing.get("weight") or 70.0),
                step=0.5,
            )

        with col_b:
            activity_level = st.selectbox(
                "Activity level *",
                ACTIVITY_CHOICES,
                index=_safe_index(
                    ACTIVITY_CHOICES,
                    existing.get("activity_level", "moderately_active"),
                ),
            )
            goal_choice = st.selectbox("Goal *", GOAL_CHOICES, index=goal_index)
            goal_other = ""
            if goal_choice == OTHER_SENTINEL:
                goal_other = st.text_input(
                    "Custom goal",
                    value=goal_stored if goal_stored not in GOAL_CHOICES else "",
                    placeholder="e.g. recomposition, prep for marathon, etc.",
                )

            diet_choice = st.selectbox(
                "Diet preference *", DIET_CHOICES, index=diet_index
            )
            diet_other = ""
            if diet_choice == OTHER_SENTINEL:
                diet_other = st.text_input(
                    "Custom diet preference",
                    value=diet_stored if diet_stored not in DIET_CHOICES else "",
                    placeholder="e.g. flexitarian, pescatarian, etc.",
                )

        st.divider()
        st.subheader("Health context (optional)")
        st.caption("Comma-separated. Leave blank if not applicable.")

        injuries_csv = st.text_input(
            "Injuries",
            value=_list_to_csv(existing.get("injuries")),
            placeholder="e.g. lower back, left knee",
        )
        medical_csv = st.text_input(
            "Medical conditions",
            value=_list_to_csv(existing.get("medical_conditions")),
            placeholder="e.g. hypertension, type 2 diabetes",
        )
        allergies_csv = st.text_input(
            "Allergies",
            value=_list_to_csv(existing.get("allergies")),
            placeholder="e.g. peanuts, shellfish, lactose",
        )

        submitted = st.form_submit_button(submit_label, type="primary")

    if not submitted:
        return

    # ── Resolve goal / diet final values ───────────────────────────────
    final_goal = goal_other.strip() if goal_choice == OTHER_SENTINEL else goal_choice
    final_diet = diet_other.strip() if diet_choice == OTHER_SENTINEL else diet_choice

    # ── Validation ─────────────────────────────────────────────────────
    if not full_name or len(full_name.strip()) < 2:
        st.error("Full name must be at least 2 characters.")
        return
    if goal_choice == OTHER_SENTINEL and not final_goal:
        st.error("Please type your custom goal.")
        return
    if diet_choice == OTHER_SENTINEL and not final_diet:
        st.error("Please type your custom diet preference.")
        return

    # ── Build payload ──────────────────────────────────────────────────
    # The backend's Pydantic validators will normalize strings, so we can send
    # them verbatim. Optional list fields are always sent (empty list → []).
    payload = {
        "full_name": full_name.strip(),
        "age": int(age),
        "gender": gender,
        "height": float(height),
        "weight": float(weight),
        "goal": final_goal,
        "activity_level": activity_level,
        "diet_preference": final_diet,
        "injuries": _csv_to_list(injuries_csv),
        "medical_conditions": _csv_to_list(medical_csv),
        "allergies": _csv_to_list(allergies_csv),
    }

    # ── Submit ─────────────────────────────────────────────────────────
    try:
        if is_edit:
            updated = patch("/api/profile/me", json=payload)
            st.success("✅ Profile updated.")
        else:
            updated = post("/api/profile/onboarding", json=payload)
            st.success("✅ Profile created. You're all set!")

        # Update the cache so the welcome-page redirect (or any other
        # require_profile() gate) no longer fires on the next rerun.
        auth.mark_profile_exists()

        render_response_panel(updated, "🔬 Profile response")

        # Refresh the form values with the just-returned profile on the next run.
        st.rerun()
    except ApiError as e:
        st.error(f"Save failed: {e}")


# ── Page entrypoint ──────────────────────────────────────────────────────
try:
    profile = _load_existing_profile()
except ApiError as e:
    st.error(f"Couldn't load your profile: {e}")
    profile = None

render_profile_form(profile)
