"""
Session-state auth + profile-gate helpers.

Two responsibilities:

1. Auth - JWT token lives in st.session_state for the duration of the
   browser tab. On reload or logout it's gone and the user has to log in
   again. This is deliberate for a developer tool - no localStorage /
   cookie persistence yet.

2. Profile gate - the AI agents in this app are useless without a populated
   profile (TDEE returns zeros, goal-aware logic falls through to generic
   defaults). So any page that calls an AI endpoint should use
   `require_profile()` instead of `require_auth()`. First-time users get
   bounced to the onboarding page automatically.

   To avoid hitting /api/profile/me on every page navigation, the result
   is cached in st.session_state.profile_exists. It's invalidated on
   logout and re-set to True by the Profile page after a successful save.
"""
from typing import Optional

import streamlit as st

from lib.api_client import ApiError, get, post


# Auth
def login(username: str, password: str) -> None:
    """POST /api/auth/login and store token + user_id in session state."""
    data = post(
        "/api/auth/login",
        json={"username": username, "password": password},
    )
    st.session_state["token"] = data["access_token"]
    st.session_state["user_id"] = data["user_id"]
    st.session_state["username"] = username
    # Invalidate any stale profile cache from a previous user on this tab.
    st.session_state.pop("profile_exists", None)


def register(email: str, password: str, username: Optional[str] = None) -> dict:
    """POST /api/auth/register. Does NOT auto-login - user must log in next."""
    payload = {"email": email, "password": password}
    if username:
        payload["username"] = username
    return post("/api/auth/register", json=payload)


def logout() -> None:
    """Clear all auth-related session state, including profile cache."""
    for key in ("token", "user_id", "username", "profile_exists"):
        st.session_state.pop(key, None)


def is_authenticated() -> bool:
    return bool(st.session_state.get("token"))


def require_auth() -> None:
    """Call at the top of every page that needs a logged-in user."""
    if not is_authenticated():
        st.warning("Please log in to access this page.")
        st.stop()


# Profile gate
def has_profile() -> bool:
    """
    Return True if the authenticated user already has a profile in the DB.

    Caches the answer in session state so subsequent page navigations are
    free. The cache is invalidated on logout and re-set by the Profile page
    after a successful create/update.

    Behavior under failures:
      - 404 from /api/profile/me -> returns False (no profile yet).
      - Any other error (500, network, 401) -> returns True. Safer default:
        show welcome screen rather than trap the user in a redirect loop
        pointing at a page that also can't reach the backend.
    """
    if not is_authenticated():
        return False

    cached = st.session_state.get("profile_exists")
    if cached is not None:
        return cached

    try:
        get("/api/profile/me")
        st.session_state["profile_exists"] = True
        return True
    except ApiError as e:
        if e.status_code == 404:
            st.session_state["profile_exists"] = False
            return False
        # Backend down / 5xx / auth glitch - don't lock the user out.
        return True


def mark_profile_exists() -> None:
    """
    Called by the Profile page after a successful onboarding/edit save.
    Updates the cache so the welcome redirect doesn't re-fire on the next
    rerun, and any other gated pages stop bouncing.
    """
    st.session_state["profile_exists"] = True


def require_profile() -> None:
    """
    Call at the top of any page that needs profile context (Chat, future
    Workout / Diet / Domain / Progress pages).

    Enforces login first, then redirects to the Profile page if no profile
    exists yet. The Profile page itself MUST NOT use this helper - it would
    create an infinite redirect.
    """
    require_auth()
    if not has_profile():
        st.switch_page("pages/6_Profile.py")
