"""
Session-state auth + profile-gate helpers.

Two responsibilities:

1. Auth - JWT token lives in st.session_state for the duration of the
   browser tab. On reload or logout it's gone and the user has to log in
   again. This is deliberate for a developer tool - no localStorage /
   cookie persistence yet.

2. Profile gate + cache - the AI agents in this app are useless without
   a populated profile (TDEE returns zeros, goal-aware logic falls through
   to generic defaults). Any page that calls an AI endpoint should use
   `require_profile()` instead of `require_auth()`. First-time users get
   bounced to the onboarding page automatically.

   Both the existence boolean AND the full profile dict are cached in
   st.session_state so:
     - has_profile() doesn't hit /api/profile/me on every nav
     - the dashboard can read profile_data directly with no API call
   Cache is invalidated on logout and replaced by the Profile page after
   a successful save.
"""
from typing import Optional

import streamlit as st

from lib.api_client import ApiError, get, post


# Auth
def login(email: str, password: str) -> None:
    """POST /api/auth/login and store token + user_id in session state."""
    data = post(
        "/api/auth/login",
        json={"email": email, "password": password},
    )
    st.session_state["token"] = data["access_token"]
    st.session_state["user_id"] = data["user_id"]
    st.session_state["email"] = email
    # Invalidate any stale caches from a previous user on this tab.
    st.session_state.pop("profile_exists", None)
    st.session_state.pop("profile_data", None)


def register(email: str, password: str, username: Optional[str] = None) -> dict:
    """POST /api/auth/register. Does NOT auto-login - user must log in next."""
    payload = {"email": email, "password": password}
    if username:
        payload["username"] = username
    return post("/api/auth/register", json=payload)


def logout() -> None:
    """Clear all auth-related session state, including profile caches."""
    for key in ("token", "user_id", "username", "profile_exists", "profile_data"):
        st.session_state.pop(key, None)


def is_authenticated() -> bool:
    return bool(st.session_state.get("token"))


def require_auth() -> None:
    """Call at the top of every page that needs a logged-in user."""
    if not is_authenticated():
        st.warning("Please log in to access this page.")
        st.stop()


# Profile gate + cache
def has_profile() -> bool:
    """
    Return True if the authenticated user already has a profile in the DB.

    On first call, fetches /api/profile/me and caches BOTH the existence
    flag AND the full profile dict for the dashboard to read for free.

    Behavior under failures:
      - 404 -> returns False (no profile yet).
      - Any other error -> returns True. Safer default: show welcome and
        let real errors surface elsewhere, rather than trap the user in a
        redirect loop pointing at a page that also can't reach the backend.
    """
    if not is_authenticated():
        return False

    cached = st.session_state.get("profile_exists")
    if cached is not None:
        return cached

    try:
        data = get("/api/profile/me")
        st.session_state["profile_exists"] = True
        st.session_state["profile_data"] = data
        return True
    except ApiError as e:
        if e.status_code == 404:
            st.session_state["profile_exists"] = False
            return False
        return True


def get_profile_data() -> Optional[dict]:
    """
    Return the cached profile dict, fetching it on first call.
    Returns None if not authenticated or the profile doesn't exist.
    Safe to call from any page after require_auth() / require_profile().
    """
    if not is_authenticated():
        return None
    if "profile_data" in st.session_state:
        return st.session_state["profile_data"]
    # Trigger a fetch+cache via has_profile().
    if has_profile():
        return st.session_state.get("profile_data")
    return None


def mark_profile_exists(data: Optional[dict] = None) -> None:
    """
    Called by the Profile page after a successful save. Updates the cache
    so the welcome redirect doesn't re-fire and so the dashboard sees the
    fresh values immediately on the next rerun.
    """
    st.session_state["profile_exists"] = True
    if data is not None:
        st.session_state["profile_data"] = data


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
