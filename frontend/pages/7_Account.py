"""
Phase 1 — Account page.

Two sections:
1. **Feedback** — pulls /api/feedback/summary + /api/feedback/history and
   renders satisfaction metrics + recent thumbs-up/down entries.
2. **Danger zone — delete account** — DELETE /api/auth/account with the
   password-confirmation body we built in the auth module. On success, the
   user is logged out and bounced back to the login page.
"""
import streamlit as st

import pandas as pd

from lib import auth
from lib.api_client import ApiError, delete, get
from lib.debug import render_response_panel

st.set_page_config(page_title="Account · AI Fitness Gym", page_icon="⚙️", layout="wide")
auth.require_auth()

st.title("⚙️ Account")
st.caption(
    f"Logged in as **{st.session_state.get('email', '—')}** "
    f"(`{st.session_state.get('user_id')}`)"
)


# ──────────────────────────────────────────────────────────────────────
# 1. FEEDBACK
# ──────────────────────────────────────────────────────────────────────
st.header("📊 My feedback")

col_summary, col_history = st.columns([1, 2])

with col_summary:
    st.subheader("Satisfaction summary")
    try:
        summary = get("/api/feedback/summary")
    except ApiError as e:
        st.error(f"Could not load summary: {e}")
        summary = None

    if summary:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total ratings", summary.get("total", 0))
        c2.metric("👍 Up", summary.get("thumbs_up", 0))
        c3.metric("👎 Down", summary.get("thumbs_down", 0))

        rate = summary.get("satisfaction_rate", 0.0) or 0.0
        st.metric("Satisfaction rate", f"{rate:.1f}%")

        recent = summary.get("recent_comments") or []
        if recent:
            st.markdown("**Recent comments**")
            for c in recent:
                st.markdown(f"> {c}")
        else:
            st.caption("No comments yet.")

        render_response_panel(summary, "🔬 Summary response")

with col_history:
    st.subheader("Recent feedback")
    limit = st.slider("How many entries", 5, 100, 20, step=5, key="fb_limit")
    try:
        history = get(f"/api/feedback/history?limit={limit}")
    except ApiError as e:
        st.error(f"Could not load history: {e}")
        history = []

    if history:
        # Flatten into a small DataFrame view.
        rows = []
        for entry in history:
            rows.append(
                {
                    "When": entry.get("created_at", ""),
                    "Rating": "👍" if entry.get("rating") == "up" else "👎",
                    "Intents": entry.get("agent_intents") or "—",
                    "Message": (entry.get("user_message") or "")[:80],
                    "Comment": (entry.get("comment") or "")[:80],
                }
            )
        st.dataframe(
            pd.DataFrame(rows),
            hide_index=True,
            use_container_width=True,
        )
        render_response_panel(history, "🔬 History response")
    else:
        st.info("No feedback submitted yet. Use the thumbs buttons on AI responses (Phase 4).")


# ──────────────────────────────────────────────────────────────────────
# 2. DANGER ZONE — DELETE ACCOUNT
# ──────────────────────────────────────────────────────────────────────
st.divider()
st.header("🚨 Danger zone")

with st.expander("Delete my account", expanded=False):
    st.error(
        "This will **permanently** delete your account, profile, feedback history, "
        "and all conversation memory in Redis. This action **cannot be undone**."
    )

    with st.form("delete_account_form", clear_on_submit=True):
        password = st.text_input(
            "Confirm your password",
            type="password",
            autocomplete="current-password",
            help="Re-authentication is required even though you're logged in.",
        )
        confirmed = st.checkbox(
            "I understand this is permanent and cannot be undone."
        )
        submit = st.form_submit_button("Delete my account", type="primary")

    if submit:
        if not password:
            st.error("Password is required to confirm deletion.")
        elif not confirmed:
            st.error("Please tick the confirmation checkbox.")
        else:
            try:
                result = delete(
                    "/api/auth/account", json={"password": password}
                )
                render_response_panel(result, "🔬 Delete response")
                # Drop the local session — the JWT is now unusable anyway since
                # the user row is gone (get_current_user re-checks DB existence).
                auth.logout()
                st.success(
                    "✅ Your account has been deleted. You've been logged out."
                )
                st.balloons()
                # Bounce the user back to the main page (which will show the
                # login form because is_authenticated() is now False).
                st.switch_page("streamlit_app.py")
            except ApiError as e:
                if e.status_code == 401:
                    st.error("Incorrect password. Account NOT deleted.")
                elif e.status_code == 404:
                    st.error(
                        "Account no longer exists. Logging you out."
                    )
                    auth.logout()
                    st.rerun()
                else:
                    st.error(f"Delete failed: {e}")
