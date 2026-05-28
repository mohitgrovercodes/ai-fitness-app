"""
Session-state auth helpers.

The JWT token lives in st.session_state for the duration of the browser tab.
On reload (or logout), it's gone and the user has to log in again. This is
deliberate for a developer tool — no localStorage/cookie persistence yet.
"""
from typing import Optional

import streamlit as st

from lib.api_client import post


def login(username: str, password: str) -> None:
    """POST /api/auth/login and store token + user_id in session state."""
    data = post(
        "/api/auth/login",
        json={"username": username, "password": password},
    )
    # success() data shape: {"access_token", "token_type", "user_id"}
    st.session_state["token"] = data["access_token"]
    st.session_state["user_id"] = data["user_id"]
    st.session_state["username"] = username


def register(email: str, password: str, username: Optional[str] = None) -> dict:
    """POST /api/auth/register. Does NOT auto-login — user must log in next."""
    payload = {"email": email, "password": password}
    if username:
        payload["username"] = username
    return post("/api/auth/register", json=payload)


def logout() -> None:
    """Clear all auth-related session state."""
    for key in ("token", "user_id", "username"):
        st.session_state.pop(key, None)


def is_authenticated() -> bool:
    return bool(st.session_state.get("token"))


def require_auth() -> None:
    """Call at the top of every protected page."""
    if not is_authenticated():
        st.warning("Please log in to access this page.")
        st.stop()
