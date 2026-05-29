"""
Phase 2.2 - Chat page (text-only).

Multi-turn conversation with the full LangGraph agent system at /api/ai/chat.
History lives in st.session_state.chat_messages and survives reruns within
the same browser session (lost on tab close or logout).

Each user turn hits the full pipeline:
  safety_guardrail -> orchestrator -> agent_router -> specialists_node
  (Training / Nutrition / Vision / Progress / Domain in parallel)
  -> synthesis_layer -> output_safety

So an assistant response may contain any combination of:
  - synthesized text answer
  - structured workout plan (cards + embedded GIFs)
  - structured meal plan (cards + macro metrics)
  - intent badges showing which agents fired

Image uploads -> /api/ai/chat-vision will be added in Step 2.3.
"""
import streamlit as st

from lib import auth
from lib.api_client import ApiError, post
from lib.debug import is_debug
from lib.renderers import render_agent_response

st.set_page_config(
    page_title="Chat - AI Fitness Gym",
    page_icon="💬",
    layout="wide",
)
# Chat needs profile context (TDEE, goal, diet preference, injuries) for the
# AI agents to give personalized answers. require_profile() enforces login
# AND redirects first-time users to the onboarding form.
auth.require_profile()

st.title("💬 Chat with your AI Coach")
st.caption(
    "Multi-turn conversation routed through the full LangGraph agent system: "
    "safety -> orchestrator -> parallel specialists (Training / Nutrition / Vision / "
    "Progress / Domain) -> synthesis."
)


# Session state initialization
# chat_messages is a list of:
#   - {"role": "user", "content": str}
#   - {"role": "assistant", "data": dict}          # unwrapped API response
#   - {"role": "assistant", "error": str}          # ApiError fallback
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []


# Sidebar - chat controls
with st.sidebar:
    st.markdown("### 💬 Chat controls")
    if st.button("🗑️ Clear chat history", use_container_width=True):
        st.session_state.chat_messages = []
        st.rerun()
    st.caption(f"Messages in thread: **{len(st.session_state.chat_messages)}**")

    st.divider()
    st.markdown("### 📷 Vision Agent (Food Only)")
    st.caption(
        "Upload an image of your food/meals to identify it, check safety, "
        "and calculate nutritional information automatically."
    )
    uploaded_file = st.file_uploader(
        "Upload food image",
        type=["png", "jpg", "jpeg", "webp"],
        key="chat_food_image",
    )
    if uploaded_file:
        st.image(uploaded_file, caption="Preview", use_container_width=True)


# Helpers
def _render_assistant_message(msg: dict) -> None:
    """Render one assistant message (response data or error fallback)."""
    if "error" in msg:
        st.error(msg["error"])
        return
    data = msg.get("data") or {}
    render_agent_response(data)
    if is_debug():
        with st.expander("🔬 Raw API response", expanded=False):
            st.json(data)


# 1. Render existing history (everything sent before this rerun)
if not st.session_state.chat_messages:
    st.info(
        "👋 Start a conversation. Try **\"Build me a 5-day weight-loss plan\"** "
        "or **\"How many calories in an apple?\"** or **\"What's my BMI?\"**"
    )

for msg in st.session_state.chat_messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
            if msg.get("image_bytes"):
                st.image(msg["image_bytes"], caption="Uploaded Image", width=300)
        else:
            _render_assistant_message(msg)


# 2. Handle new input
prompt = st.chat_input(
    "Ask about workouts, nutrition, your goals, or anything fitness-related..."
)

if prompt:
    # 2a. Persist + render the user turn
    user_turn = {"role": "user", "content": prompt}
    image_bytes = None
    if uploaded_file:
        image_bytes = uploaded_file.getvalue()
        user_turn["image_bytes"] = image_bytes
        
    st.session_state.chat_messages.append(user_turn)
    
    with st.chat_message("user"):
        st.markdown(prompt)
        if image_bytes:
            st.image(image_bytes, caption="Uploaded Image", width=300)

    # 2b. Call backend, render assistant turn live
    with st.chat_message("assistant"):
        with st.spinner("Coach is thinking..."):
            try:
                if image_bytes:
                    data = post(
                        "/api/ai/chat-vision",
                        data={"message": prompt},
                        files={"file": (uploaded_file.name, image_bytes, uploaded_file.type)},
                        timeout=3600,
                    )
                else:
                    data = post(
                        "/api/ai/chat",
                        json={"message": prompt, "context": {}},
                        timeout=3600,
                    )
                st.session_state.chat_messages.append(
                    {"role": "assistant", "data": data}
                )
                render_agent_response(data)
                if is_debug():
                    with st.expander("🔬 Raw API response", expanded=False):
                        st.json(data)
            except ApiError as e:
                error_msg = f"Coach is unavailable: {e}"
                st.error(error_msg)
                st.session_state.chat_messages.append(
                    {"role": "assistant", "error": error_msg}
                )
