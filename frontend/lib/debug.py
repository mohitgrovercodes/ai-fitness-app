"""
Debug panel utilities.

Phase 0 ships the sidebar toggle and a generic JSON-response viewer.
Phase 5 will add per-agent trace tabs (intents, specialist breakdown, RAG hits,
timing, vision-tier decisions) once the backend ?debug=1 hook lands.
"""
from typing import Any

import streamlit as st


def debug_toggle_sidebar() -> bool:
    """Render the sidebar toggle and return the current state."""
    return st.sidebar.toggle(
        "🐛 Developer mode",
        key="debug_mode",
        help="Show raw API responses, agent traces, and timing data.",
    )


def is_debug() -> bool:
    return bool(st.session_state.get("debug_mode"))


def render_response_panel(payload: Any, title: str = "🔬 Raw API response") -> None:
    """Render a JSON response inside an expander when debug mode is on."""
    if not is_debug():
        return
    with st.expander(title, expanded=False):
        st.json(payload)
