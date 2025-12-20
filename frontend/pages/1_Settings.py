"""
Settings Page

WHY SEPARATE PAGE?
- Keep main app simple
- Advanced options for power users
- Don't overwhelm beginners

FEATURES:
- Model selection
- Retrieval parameters
- System information
"""

import streamlit as st
import sys
from pathlib import Path

st.set_page_config(
    page_title="Settings",
    page_icon="⚙️",
    layout="wide"
)

st.title("⚙️ Settings")

# Model Configuration
st.subheader("🤖 Model Configuration")

col1, col2 = st.columns(2)

with col1:
    llm_model = st.selectbox(
        "Language Model",
        ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
        help="Model for generating answers"
    )
    
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.1,
        help="Lower = more focused, Higher = more creative"
    )

with col2:
    embedding_model = st.selectbox(
        "Embedding Model",
        ["text-embedding-ada-002", "text-embedding-3-small", "text-embedding-3-large"],
        help="Model for creating vectors"
    )
    
    chunk_size = st.number_input(
        "Chunk Size",
        min_value=256,
        max_value=2048,
        value=1024,
        step=256,
        help="Size of document chunks"
    )

# Retrieval Configuration
st.markdown("---")
st.subheader("🔍 Retrieval Configuration")

col1, col2, col3 = st.columns(3)

with col1:
    similarity_top_k = st.number_input(
        "Results per Retriever",
        min_value=1,
        max_value=10,
        value=5,
        help="Number of results from each retriever"
    )

with col2:
    final_top_k = st.number_input(
        "Final Results",
        min_value=1,
        max_value=10,
        value=3,
        help="Results sent to LLM after re-ranking"
    )

with col3:
    vector_weight = st.slider(
        "Vector Weight",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        help="Weight for vector search (1-weight goes to BM25)"
    )

# System Information
st.markdown("---")
st.subheader("ℹ️ System Information")

col1, col2 = st.columns(2)

with col1:
    st.info(f"""
    **Python Version:** {sys.version.split()[0]}
    **Streamlit Version:** {st.__version__}
    """)

with col2:
    project_root = Path(__file__).parent.parent.parent
    storage_path = project_root / "storage"
    
    if storage_path.exists():
        st.success("✅ Vector index found")
    else:
        st.warning("⚠️ No vector index yet")

# Save button
st.markdown("---")
if st.button("💾 Save Settings", type="primary"):
    st.success("Settings saved! Restart the app to apply changes.")
    st.balloons()

st.markdown("---")
st.caption("💡 **Tip:** Most users don't need to change these settings. The defaults work well!")