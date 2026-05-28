"""
AI Fitness Gym — Developer Visualization
Entry point.

Behavior:
- If no JWT in st.session_state → show Login / Register tabs.
- If logged in → show welcome page describing the rest of the app.
- Sidebar always shows the 🐛 Developer mode toggle and (when logged in) a
  Logout button + the current username.
"""
import streamlit as st

from lib import auth
from lib.api_client import ApiError
from lib.debug import debug_toggle_sidebar, render_response_panel

st.set_page_config(
    page_title="AI Fitness Gym",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def render_login_form() -> None:
    st.title("🏋️ AI Fitness Gym")
    st.caption(
        "Developer visualization for the Multi-Agent Fitness & Nutrition Platform"
    )

    tab_login, tab_register = st.tabs(["Login", "Register"])

    # ── Login tab ────────────────────────────────────────────────────
    with tab_login:
        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("Username", autocomplete="username")
            password = st.text_input(
                "Password", type="password", autocomplete="current-password"
            )
            submitted = st.form_submit_button("Log in", type="primary")
            if submitted:
                if not username or not password:
                    st.error("Username and password are required.")
                else:
                    try:
                        auth.login(username, password)
                        st.success(f"Welcome back, {username}!")
                        st.rerun()
                    except ApiError as e:
                        st.error(f"Login failed: {e}")

    # ── Register tab ─────────────────────────────────────────────────
    with tab_register:
        with st.form("register_form", clear_on_submit=True):
            r_email = st.text_input("Email", autocomplete="email")
            r_username = st.text_input(
                "Username (optional — defaults to email prefix)"
            )
            r_password = st.text_input(
                "Password", type="password", autocomplete="new-password"
            )
            r_submitted = st.form_submit_button("Create account", type="primary")
            if r_submitted:
                if not r_email or not r_password:
                    st.error("Email and password are required.")
                else:
                    try:
                        result = auth.register(
                            r_email, r_password, r_username or None
                        )
                        st.success(
                            "Account created. Switch to the **Login** tab to sign in."
                        )
                        render_response_panel(result, "🔬 Register response")
                    except ApiError as e:
                        st.error(f"Registration failed: {e}")


def render_welcome() -> None:
    user = st.session_state.get("username") or st.session_state.get("user_id", "User")
    st.title(f"👋 Welcome, {user}")
    st.caption(f"User ID: `{st.session_state.get('user_id')}`")

    st.markdown(
        """
        ### What you can do here

        This is the developer visualization for the **Agentic AI Gym** multi-agent system.
        Use the sidebar to navigate between pages (more pages will appear as Phases 1–5 ship):

        - **💬 Chat** — multi-turn conversation with the full agent graph *(Phase 2)*
        - **🏋️ Workout Plan** — direct hit on the Training Agent *(Phase 3)*
        - **🥗 Diet Plan** — direct hit on the Nutrition Agent *(Phase 3)*
        - **📚 Domain Q&A** — direct hit on the Domain Agent *(Phase 3)*
        - **📈 Progress** — Progress Agent visualizer *(Phase 4)*
        - **👤 Profile** — onboarding + edit *(Phase 1)*
        - **⚙️ Account** — feedback history + delete account *(Phase 1)*

        Toggle **🐛 Developer mode** in the sidebar to surface agent traces,
        RAG hits, and raw JSON for any AI response.
        """
    )


def render_sidebar() -> None:
    debug_toggle_sidebar()
    st.sidebar.divider()
    if auth.is_authenticated():
        st.sidebar.write(
            f"**Logged in as:** {st.session_state.get('username', '—')}"
        )
        if st.sidebar.button("Logout"):
            auth.logout()
            st.rerun()
    else:
        st.sidebar.caption("Not logged in.")


# ── Main ─────────────────────────────────────────────────────────────
render_sidebar()
if auth.is_authenticated():
    render_welcome()
else:
    render_login_form()
