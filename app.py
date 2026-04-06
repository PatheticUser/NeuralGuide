import streamlit as st
import pandas as pd
from agent_core import get_agentic_models_from_cloud
from ollama_utils import get_local_models, is_ollama_running
from config import APP_TITLE, APP_TAGLINE, APP_DESCRIPTION, OPENAI_API_KEY
import random

# --- STYLES & SETUP ---
st.set_page_config(page_title=APP_TITLE, layout="wide", page_icon="🔍")

st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
    }
    .card {
        background: #1E2129;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        margin-bottom: 20px;
        border-left: 4px solid #4CAF50;
    }
    .card-title {
        color: #4CAF50;
        font-size: 24px;
        font-weight: 600;
        margin-bottom: 10px;
    }
    .card-parameter {
        font-size: 14px;
        color: #B2B8C2;
        margin-bottom: 5px;
    }
    .card-content {
        color: #E2E8F0;
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)

# --- SESSION STATE ---
if "search_history" not in st.session_state:
    st.session_state.search_history = []

# --- SIDEBAR ---
with st.sidebar:
    st.title("⚙️ System Status")
    
    st.subheader("Connectivity")
    openai_status = "🟢 Connected" if OPENAI_API_KEY else "🔴 Missing Key"
    st.markdown(f"**OpenAI API:** {openai_status}")
    
    ollama_status = "🟢 Connected" if is_ollama_running() else "🔴 Offline"
    st.markdown(f"**Ollama Local:** {ollama_status}")
    
    st.subheader("📚 Search History")
    if st.session_state.search_history:
        for i, q in enumerate(reversed(st.session_state.search_history)):
            st.caption(f"- {q}")
            if i >= 4:
                break
    else:
        st.caption("No recent searches.")
        
    st.markdown("---")
    st.subheader("ℹ️ How it works")
    st.info(
        "AgentLens acts as an AI discovery tool. "
        "Enter a specific agent use case, and it will fetch recommendations "
        "from Cloud Models (via OpenAI Responses API) and check local availability (via Ollama)."
    )

# --- MAIN PAGE ---
st.title(f"🔍 {APP_TITLE}")
st.subheader(APP_TAGLINE)
st.write(APP_DESCRIPTION)

st.markdown("---")

query = st.text_input("Describe your agentic use case (e.g., 'coding assistant with parallel tool calling'):")

if st.button("Search Models", type="primary"):
    if not query:
        st.warning("Please enter a use case query.")
    else:
        st.session_state.search_history.append(query)
        
        with st.spinner("Analyzing models..."):
            cloud_models = get_agentic_models_from_cloud(query)
            local_models = get_local_models() 
            
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.header("🌩️ Cloud Recommendations")
            if not cloud_models or "Error" in cloud_models[0].get("Model Name", ""):
                st.error("Failed to fetch cloud models or missing API Key.")
            else:
                for idx, model in enumerate(cloud_models):
                    with st.expander(f"✨ {model.get('Model Name')} - Expand for Details", expanded=(idx==0)):
                        st.markdown(f"<div class='card-title'>{model.get('Model Name')}</div>", unsafe_allow_html=True)
                        st.markdown(f"<div class='card-content'><b>Description:</b> {model.get('Description')}</div>", unsafe_allow_html=True)
                        st.markdown(f"<div class='card-parameter'><b>Parameters:</b> {model.get('Parameters')}</div>", unsafe_allow_html=True)
                        st.markdown(f"<div class='card-parameter'><b>Key Features:</b> {model.get('Key Features')}</div>", unsafe_allow_html=True)
                        st.markdown(f"<div class='card-parameter'><b>Tool/Function Calling:</b> {model.get('Tool/Function Calling')}</div>", unsafe_allow_html=True)
        
        with col2:
            st.header("💻 Local Models (Ollama)")
            if not local_models:
                st.warning("No local models found on Ollama or service is offline.")
            else:
                # Build comparison table
                comparison_data = []
                for idx, t_model in enumerate(local_models + cloud_models[:2]):  # add a couple cloud models for comparison
                    is_cloud = idx >= len(local_models)
                    # mock suitability score, cost, privacy, and latency
                    score = random.randint(75, 98) if is_cloud else random.randint(60, 90)
                    cost = "$$$" if is_cloud else "Free"
                    privacy = "Low" if is_cloud else "High"
                    latency = "Medium" if is_cloud else "Low (Hardware dep.)"
                    r_type = "Cloud" if is_cloud else "Local"
                    
                    comparison_data.append({
                        "Model": t_model.get("Model Name"),
                        "Type": r_type,
                        "Suitability Score": f"{score}/100",
                        "Cost": cost,
                        "Privacy": privacy,
                        "Latency": latency
                    })
                
                df = pd.DataFrame(comparison_data)
                
                st.write("### Cloud vs. Local Comparison")
                st.dataframe(
                    df.style.applymap(
                        lambda x: 'color: green' if x == 'Free' or x == 'High' else ('color: red' if x == 'Low' or x == '$$$' else ''),
                        subset=['Cost', 'Privacy']
                    ),
                    use_container_width=True
                )
