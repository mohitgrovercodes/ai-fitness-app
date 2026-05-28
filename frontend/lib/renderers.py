"""
Reusable Streamlit renderers for AI agent responses.

This module centralizes the visual logic so every page (Chat, Workout,
Diet, Domain, Progress) calls the same functions and gets identical-looking
workout cards, meal cards, macro charts, and intent badges.

The top-level entrypoint is `render_agent_response(data)` — pass it the
unwrapped `data` dict from any backend endpoint and it figures out which
sections to render based on which keys are present.

Media handling:
- The backend stores exercise GIFs/images at paths relative to
  `Data/exercises-dataset/` (e.g. "videos/0044-XlZ4lAC.gif").
- Frontend resolves these to absolute filesystem paths under
  `<repo_root>/Data/exercises-dataset/` and hands them to `st.image()`.
- This works because the Streamlit app runs on the same machine as the
  backend (local-only deployment). For a remote deployment, the backend
  would need to expose a `/media/{path}` static route.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import streamlit as st

# ──────────────────────────────────────────────────────────────────────
# Dataset path — resolves to <repo_root>/Data/exercises-dataset/
# This file lives at <repo_root>/frontend/lib/renderers.py, so:
#   parent (lib) → parent (frontend) → parent (repo root) → Data/exercises-dataset/
# ──────────────────────────────────────────────────────────────────────
_DATASET_ROOT = (
    Path(__file__).resolve().parent.parent.parent / "Data" / "exercises-dataset"
)


def _resolve_media_path(relative_path: Optional[str]) -> Optional[Path]:
    """Return an absolute Path if the file exists on disk, else None."""
    if not relative_path:
        return None
    full = _DATASET_ROOT / relative_path
    return full if full.exists() else None


# ──────────────────────────────────────────────────────────────────────
# 1. Intent badges
# ──────────────────────────────────────────────────────────────────────
_INTENT_EMOJI = {
    "workout": "🏋️",
    "nutrition": "🥗",
    "image": "📷",
    "progress": "📈",
    "general": "💡",
    "out_of_scope": "🚧",
}


def render_intents(intents: list[str]) -> None:
    """Small caption-line chip strip showing which agents fired."""
    if not intents:
        return
    chips = [f"`{_INTENT_EMOJI.get(i, '•')} {i}`" for i in intents]
    st.caption("Agents engaged: " + " ".join(chips))


# ──────────────────────────────────────────────────────────────────────
# 2. Macro / calorie totals
# ──────────────────────────────────────────────────────────────────────
def render_daily_totals(daily: dict, title: Optional[str] = "📊 Daily totals") -> None:
    """4-column metric strip for calories + protein/carbs/fat."""
    if not daily:
        return
    if title:
        st.markdown(f"#### {title}")
    cols = st.columns(4)
    cols[0].metric("🔥 Calories", f"{daily.get('calories', '?')}")
    cols[1].metric("💪 Protein", str(daily.get("protein", "—")))
    cols[2].metric("🌾 Carbs", str(daily.get("carbs", "—")))
    cols[3].metric("🥑 Fat", str(daily.get("fat", "—")))
    note = daily.get("note")
    if note:
        st.caption(note)


def render_per_day_totals(per_day: dict) -> None:
    """Per-day totals table for multi-day plans."""
    if not per_day:
        return
    import pandas as pd

    rows = []
    for day, totals in per_day.items():
        rows.append(
            {
                "Day": day,
                "Calories": totals.get("calories", "?"),
                "Protein": totals.get("protein", "—"),
                "Carbs": totals.get("carbs", "—"),
                "Fat": totals.get("fat", "—"),
            }
        )
    st.markdown("#### 📅 Per-day breakdown")
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────
# 3. Workout
# ──────────────────────────────────────────────────────────────────────
def _render_single_exercise(
    ex: dict, gifs: dict, images: dict
) -> None:
    """One exercise as a 2-column card: media on left, details on right."""
    name = ex.get("name") or "Exercise"

    cols = st.columns([1, 2])

    # ── Left: media ─────────────────────────────────────────────
    with cols[0]:
        gif_rel = ex.get("gif_path") or gifs.get(name)
        img_rel = ex.get("image_path") or images.get(name)

        gif_path = _resolve_media_path(gif_rel)
        img_path = _resolve_media_path(img_rel)

        if gif_path:
            st.image(str(gif_path), use_container_width=True)
        elif img_path:
            st.image(str(img_path), use_container_width=True)
        else:
            st.caption("📷 _No media available_")

    # ── Right: details ──────────────────────────────────────────
    with cols[1]:
        st.markdown(f"**{name}**")

        muscles = ex.get("target_muscle") or []
        if muscles:
            if isinstance(muscles, list):
                muscles_str = ", ".join(str(m) for m in muscles)
            else:
                muscles_str = str(muscles)
            st.caption(f"🎯 {muscles_str}")

        meta = st.columns(2)
        sets_val = ex.get("sets", "")
        reps_val = ex.get("reps", "")
        if sets_val:
            meta[0].markdown(f"**Sets:** {sets_val}")
        if reps_val:
            meta[1].markdown(f"**Reps:** {reps_val}")

        benefit = ex.get("benefit", "")
        if benefit:
            st.markdown(f"💡 *{benefit}*")

        description = ex.get("description", "")
        if description:
            with st.expander("How to do it"):
                st.write(description)


def render_workout_cards(
    workout: list[dict],
    gifs: Optional[dict] = None,
    images: Optional[dict] = None,
) -> None:
    """Render the workout list, grouped by `day` field."""
    if not workout:
        return

    gifs = gifs or {}
    images = images or {}

    # Group by day; preserve insertion order (Python 3.7+ dict guarantees it).
    by_day: dict[str, list[dict]] = {}
    for ex in workout:
        day = (ex.get("day") or "").strip() or "Today's Workout"
        by_day.setdefault(day, []).append(ex)

    for day, exercises in by_day.items():
        with st.container(border=True):
            st.markdown(f"#### {day}")
            for i, ex in enumerate(exercises):
                _render_single_exercise(ex, gifs, images)
                if i < len(exercises) - 1:
                    st.divider()


def render_rest_days(rest_days: list[dict]) -> None:
    """Render rest day entries (separate from active workout exercises)."""
    if not rest_days:
        return
    for rest in rest_days:
        with st.container(border=True):
            st.markdown(f"#### 😴 {rest.get('day', 'Rest Day')}")
            benefit = rest.get("benefit", "")
            if benefit:
                st.markdown(f"💡 *{benefit}*")
            description = rest.get("description", "")
            if description:
                st.write(description)


# ──────────────────────────────────────────────────────────────────────
# 4. Meals
# ──────────────────────────────────────────────────────────────────────
def _render_single_meal(meal: dict) -> None:
    """One meal entry as a 2-column row: type/portion on left, details on right."""
    cols = st.columns([1, 3])

    with cols[0]:
        meal_type = meal.get("type") or "Meal"
        st.markdown(f"**{meal_type}**")
        portion = meal.get("portion", "")
        if portion:
            st.caption(f"📏 {portion}")

    with cols[1]:
        name = meal.get("name") or "—"
        st.markdown(f"**{name}**")

        macros = st.columns(4)
        macros[0].markdown(f"🔥 **{meal.get('calories', 0)}** kcal")
        macros[1].markdown(f"💪 {meal.get('protein', '—')}")
        macros[2].markdown(f"🌾 {meal.get('carbs', '—')}")
        macros[3].markdown(f"🥑 {meal.get('fat', '—')}")

        benefit = meal.get("benefit", "")
        if benefit:
            st.caption(f"💡 {benefit}")


def render_meal_cards(meals: list[dict]) -> None:
    """Render meals grouped by day; single-day plans get a default header."""
    if not meals:
        return

    by_day: dict[str, list[dict]] = {}
    for meal in meals:
        day = (meal.get("day") or "").strip() or "Today's Meals"
        by_day.setdefault(day, []).append(meal)

    for day, day_meals in by_day.items():
        with st.container(border=True):
            st.markdown(f"#### 🍽️ {day}")
            for i, meal in enumerate(day_meals):
                _render_single_meal(meal)
                if i < len(day_meals) - 1:
                    st.divider()


# ──────────────────────────────────────────────────────────────────────
# 5. Plan-level extras (summary, tip)
# ──────────────────────────────────────────────────────────────────────
def render_plan_summary(summary: str) -> None:
    """Direct-API endpoints include a top-level `summary` field."""
    if not summary:
        return
    with st.container(border=True):
        st.markdown("**Plan Overview**")
        st.markdown(summary)


def render_plan_tip(tip: str) -> None:
    """Closing tip from the agent."""
    if not tip:
        return
    st.info(f"💡 **Tip:** {tip}")


# ──────────────────────────────────────────────────────────────────────
# 6. Top-level dispatcher
# ──────────────────────────────────────────────────────────────────────
def render_agent_response(data: Any) -> None:
    """
    Render any AI response payload. Tolerates missing keys and routes to
    the appropriate sub-renderer for whatever structured data is present.

    Accepts the unwrapped `data` dict from /api/ai/chat, /generate-workout,
    /generate-diet, or /ask-domain.
    """
    if not isinstance(data, dict):
        if data:
            st.markdown(str(data))
        return

    # 1. Plan summary (direct-API endpoints)
    render_plan_summary(data.get("summary", ""))

    # 2. Main narrative text (chat) — keys: response, answer
    text = data.get("response") or data.get("answer", "")
    if text:
        st.markdown(text)

    # 3. Intents
    render_intents(data.get("intents", []))

    # 4. Workout block
    workout = data.get("workout") or []
    if workout:
        st.markdown("### 🏋️ Workout")
        render_workout_cards(
            workout,
            gifs=data.get("exercise_gifs"),
            images=data.get("exercise_images"),
        )

    # 5. Rest days
    rest_days = data.get("rest_days") or []
    if rest_days:
        render_rest_days(rest_days)

    # 6. Meal plan
    meals = data.get("meals") or []
    if meals:
        st.markdown("### 🥗 Meal Plan")
        render_meal_cards(meals)

    # 7. Macro totals
    render_daily_totals(data.get("daily_totals") or {})
    render_per_day_totals(data.get("per_day_totals") or {})

    # 8. Closing tip
    render_plan_tip(data.get("tip", ""))

    # 9. Sources (Domain agent)
    sources = data.get("sources")
    if sources:
        st.caption(f"📚 Sources: {sources}")
