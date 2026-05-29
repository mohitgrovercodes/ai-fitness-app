"""
Phase 3 - Standalone Research & Domain Q&A.

Directly queries the Domain Agent endpoint `/api/ai/ask-domain`
to search verified fitness and nutrition textbook databases.
"""
import streamlit as st

from lib import auth
from lib.api_client import ApiError, post
from lib.debug import is_debug
from lib.renderers import render_agent_response

st.set_page_config(
    page_title="Domain Q&A - AI Fitness Gym",
    page_icon="📚",
    layout="wide",
)

# Enforce profile context
auth.require_profile()

st.title("📚 Research Database Q&A")
st.caption(
    "Directly query our verified clinical database of fitness and nutrition textbooks. "
    "This bypassing the main chat loop to perform rapid, exact evidence lookups."
)

st.markdown("### 🔍 Enter your fitness or nutrition research question")
st.write("Ask questions like: *\"What is muscle hypertrophy?\"*, *\"Explain the science behind creatine loading.\"*, or *\"What are the physiological benefits of high-intensity interval training?\"*")

# Simple search form
with st.form("domain_query_form", clear_on_submit=False):
    query = st.text_input(
        "Search Query",
        placeholder="Type your scientific question here...",
        help="Type a question backed by sports science and clinical nutrition.",
    )
    submitted = st.form_submit_button("Search Research Database", type="primary", use_container_width=True)

# Handle search submission
if submitted:
    if not query.strip():
        st.error("Please enter a valid search query.")
    else:
        with st.spinner("Searching the sports science library..."):
            try:
                data = post(
                    "/api/ai/ask-domain",
                    json={"message": query.strip()},
                    timeout=3600,
                )
                
                st.success("🔬 Evidence retrieval completed!")
                
                # Render returned response and textbook citations
                render_agent_response(data)
                
                if is_debug():
                    with st.expander("🔬 Raw API Response", expanded=False):
                        st.json(data)
                        
            except ApiError as e:
                st.error(f"Failed to query the database: {e}")
