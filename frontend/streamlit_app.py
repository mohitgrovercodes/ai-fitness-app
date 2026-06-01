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

username = st.text_input("username", autocomplete="username")            
def render_login_form() -> None:
    token = st.query_params.get("token")
    if token:
        st.title("🔐 Reset Password")
        with st.form("reset_password_form", clear_on_submit=True):
            new_password = st.text_input("New Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            if st.form_submit_button("Reset Password", type="primary"):
                if not new_password or new_password != confirm_password:
                    st.error("Passwords do not match or are empty!")
                else:
                    try:
                        auth.reset_password(token, new_password)
                        st.success("Password reset successfully! You can now log in.")
                        st.query_params.clear()
                    except ApiError as e:
                        st.error(f"Reset failed: {e}")
        if st.button("Back to Login"):
            st.query_params.clear()
            st.rerun()
        return

    st.title("🏋️ AI Fitness Gym")
    st.caption(
        "Developer visualization for the Multi-Agent Fitness & Nutrition Platform"
    )

    tab_login, tab_register = st.tabs(["Login", "Register"])

    # ── Login tab ────────────────────────────────────────────────────
    with tab_login:
        with st.form("login_form", clear_on_submit=False):
            email = st.text_input("email", autocomplete="email")
            password = st.text_input(
                "Password", type="password", autocomplete="current-password"
            )
            submitted = st.form_submit_button("Log in", type="primary")
            if submitted:
                if not email or not password:
                    st.error("Username and password are required.")
                else:
                    try:
                        auth.login(email, password)
                        st.success(f"Welcome back, {username}!")
                        st.rerun()
                    except ApiError as e:
                        st.error(f"Login failed: {e}")
                        
        with st.expander("Forgot Password?"):
            with st.form("forgot_password_form"):
                fp_email = st.text_input("Enter your registered email")
                if st.form_submit_button("Send Reset Link"):
                    if not fp_email:
                        st.error("Email is required.")
                    else:
                        try:
                            auth.forgot_password(fp_email)
                            st.success("If that email exists, a reset link has been sent (check console for link).")
                        except ApiError as e:
                            st.error(f"Failed: {e}")

    # ── Register tab ─────────────────────────────────────────────────
    with tab_register:
        with st.form("register_form", clear_on_submit=True):
            r_email = st.text_input("Email", autocomplete="email")
            r_username = st.text_input(
                "Username"
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
    profile = auth.get_profile_data() or {}
    
    # ── Map Goal choices to clean reader-friendly text ─────────────────
    goal = profile.get("goal")
    goal_map = {
        "fat_loss": "Fat Loss",
        "muscle_gain": "Muscle Gain",
        "maintenance": "Maintenance",
        "athletic_performance": "Athletic Performance"
    }
    goal_str = goal_map.get(str(goal).lower(), str(goal).title() if goal else "Not Set")

    # ── Map Activity choices to clean reader-friendly text ─────────────
    activity_level = profile.get("activity_level")
    activity_map = {
        "sedentary": "Sedentary",
        "lightly_active": "Lightly Active",
        "moderately_active": "Moderately Active",
        "very_active": "Very Active",
        "extra_active": "Extra Active"
    }
    activity_str = activity_map.get(str(activity_level).lower(), str(activity_level).title() if activity_level else "—")

    # ── Dynamic BMI Calculation and Health Classification ──────────────
    weight = profile.get("weight")
    height = profile.get("height")
    bmi_str = "—"
    if weight and height:
        try:
            h_m = float(height) / 100.0
            bmi_val = float(weight) / (h_m * h_m)
            
            if bmi_val < 18.5:
                bmi_class = " (Underweight)"
            elif bmi_val < 25.0:
                bmi_class = " (Normal)"
            elif bmi_val < 30.0:
                bmi_class = " (Overweight)"
            else:
                bmi_class = " (Obese)"
            bmi_str = f"{bmi_val:.1f}{bmi_class}"
        except Exception:
            pass

    # ── 1. HERO SECTION ──────────────────────────────────────────────
    display_name = profile.get("full_name") or st.session_state.get("username") or "User"
    st.title(f"👋 Welcome back, {display_name}!")
    st.subheader(f"🎯 Goal: :green[{goal_str}]")
    st.write("")

    # ── 2. BIOMETRIC METRICS STRIP ──────────────────────────────────
    st.markdown("### 📊 My Biometrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🏋️ Weight", f"{weight} kg" if weight else "—")
    c2.metric("📏 Height", f"{height} cm" if height else "—")
    c3.metric("🧠 BMI", bmi_str)
    c4.metric("🏃 Activity", activity_str)
    
    st.divider()

    # ── 3. 2x3 FEATURE GRID ──────────────────────────────────────────
    st.markdown("### 🚀 Platform Features")
    
    # Row 1
    r1_col1, r1_col2, r1_col3 = st.columns(3)
    
    with r1_col1:
        with st.container(border=True):
            st.subheader("💬 Chat")
            st.caption("Phase 2 · Active")
            st.write("Multi-turn conversation with the full AI coach agent graph.")
            if st.button("Launch Chat", type="primary", use_container_width=True, key="btn_chat"):
                st.switch_page("pages/3_Chat.py")
                
    with r1_col2:
        with st.container(border=True):
            st.subheader("🏋️ Workout Plan")
            st.caption("Phase 3 · Active")
            st.write("Direct access to the Training Agent to generate targeted workouts.")
            if st.button("Generate Workout", type="primary", use_container_width=True, key="btn_workout"):
                st.switch_page("pages/4_Workout.py")
            
    with r1_col3:
        with st.container(border=True):
            st.subheader("🥗 Diet Plan")
            st.caption("Phase 3 · Active")
            st.write("Direct access to the Nutrition Agent to generate customized diets.")
            if st.button("Generate Diet", type="primary", use_container_width=True, key="btn_diet"):
                st.switch_page("pages/5_Diet.py")

    # Row 2
    r2_col1, r2_col2, r2_col3 = st.columns(3)
    
    with r2_col1:
        with st.container(border=True):
            st.subheader("📚 Domain Q&A")
            st.caption("Phase 3 · Active")
            st.write("Direct Q&A with our Fitness & Nutrition research book database.")
            if st.button("Query Database", type="primary", use_container_width=True, key="btn_domain"):
                st.switch_page("pages/8_Domain.py")
            
    with r2_col2:
        with st.container(border=True):
            st.subheader("📈 Progress")
            st.caption(":orange[Phase 4 · Locked]")
            st.write("Visual analytics and progress summaries of your fitness journey.")
            st.button("Locked", disabled=True, use_container_width=True, key="btn_progress")
            
    with r2_col3:
        with st.container(border=True):
            st.subheader("👤 Profile")
            st.caption("Phase 1 · Active")
            st.write("View or update your age, weight, goals, activity level, and medical context.")
            if st.button("Edit Profile", use_container_width=True, key="btn_profile"):
                st.switch_page("pages/6_Profile.py")
                
    st.divider()

    # ── 4. BOTTOM UTILITY ROW ──────────────────────────────────────
    st.markdown("### ⚙️ Utilities")
    u_col1, u_col2 = st.columns(2)
    with u_col1:
        if st.button("⚙️ Manage Account", use_container_width=True, key="btn_account"):
            st.switch_page("pages/7_Account.py")
    with u_col2:
        if st.button("🛑 Logout", type="secondary", use_container_width=True, key="btn_logout"):
            auth.logout()
            st.rerun()


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
    # First-time login: if the user has no profile yet, route them straight
    # to the onboarding form. The Profile page itself uses require_auth()
    # (not require_profile()) so this redirect can't loop. After the user
    # saves their profile, the cache is set to True and they land here on
    # subsequent visits.
    if not auth.has_profile():
        st.switch_page("pages/6_Profile.py")
    render_welcome()
else:
    render_login_form()
