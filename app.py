import streamlit as st
from utils.config import APP_TITLE, APP_ICON
from utils.styles import apply_styling

# Configure the app page configuration
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply global styles
apply_styling()

st.title("Welcome to EMIPredict AI 💎")

st.markdown(
    """
    ### Intelligent Financial Risk Assessment Platform
    
    Please navigate using the sidebar to explore the application:
    
    - **🏠 Home**: Project Overview & Problem Statement
    - **🧮 EMI Calculator**: Real-time Eligibility & Prediction
    - **📊 Model Performance**: Trust Meter & Metrics
    - **📈 Data Insights**: Visual Patterns & Trends
    - **🤖 AI Assistant**: Project Chatbot
    
    ---
    *Built with Streamlit & Machine Learning*
    """
)

# Sidebar Info
st.sidebar.info("Select a page above to begin.")
st.sidebar.markdown("---")
st.sidebar.write("Developed for FinTech AI Assessment")
