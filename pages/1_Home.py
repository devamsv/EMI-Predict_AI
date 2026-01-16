import streamlit as st
from utils.styles import apply_styling, render_hero_section

# Config
st.set_page_config(page_title="Home - EMIPredict AI", page_icon="🏠", layout="wide")
apply_styling()

# Hero Section
render_hero_section("EMIPredict AI", "Intelligent Financial Risk Assessment Platform")

# Introduction Section
st.markdown("### 🏦 Transforming Financial Decision Making")
st.markdown(
    """
    <div style="font-size: 1.05rem; line-height: 1.6; color: #374151; margin-bottom: 2rem;">
        In the rapidly evolving landscape of <b>FinTech</b>, accurately assessing loan eligibility and repayment capacity 
        is critical for both lenders and borrowers. <b>EMIPredict AI</b> bridges the gap between complex financial data 
        and actionable insights using state-of-the-art machine learning algorithms.
        <br><br>
        This platform provides a robust, real-time interface for evaluating credit risk, predicting safe EMI limits, 
        and offering personalized financial recommendations.
    </div>
    """, 
    unsafe_allow_html=True
)

# Problem & Solution
col1, col2 = st.columns(2)

with col1:
    st.markdown(
        """
        <div class="card" style="height: 100%;">
            <h3 style="color: #1e40af; margin-bottom: 12px;">🚧 The Business Challenge</h3>
            <p style="font-size: 1rem; line-height: 1.6; color: #4b5563;">
                <b>High Default Rates:</b> Traditional credit scoring often misses behavioral nuances, leading to potential defaults.<br>
                <b>Opaque Processes:</b> Borrowers frequently struggle to understand why their applications are rejected.<br>
                <b>Manual Bottle-necks:</b> Manual underwriting is slow, error-prone, and unscalable for high-volume lending.
            </p>
        </div>
        """, 
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        """
        <div class="card" style="height: 100%;">
            <h3 style="color: #15803d; margin-bottom: 12px;">🚀 The AI Solution</h3>
            <p style="font-size: 1rem; line-height: 1.6; color: #4b5563;">
                <b>Automated Risk Assessment:</b> Instant eligibility checks using an optimized XGBoost classifier.<br>
                <b>Precision Forecasting:</b> Accurate 'Max Safe EMI' prediction via Random Forest regression.<br>
                <b>Explainable AI:</b> Transparent reasoning behind every decision to build user trust.
            </p>
        </div>
        """, 
        unsafe_allow_html=True
    )

# Key Capabilities
st.markdown("### ⚡ Key Capabilities")
cap_col1, cap_col2, cap_col3 = st.columns(3)

with cap_col1:
    st.markdown(
        """
        <div class="card" style="text-align:center;">
            <div style="font-size: 2rem; margin-bottom: 10px;">🛡️</div>
            <h4 style="margin-bottom: 8px;">Risk Classification</h4>
            <p style="font-size: 0.9rem; color: #64748b;">Detect high-risk profiles instantly with 94% accuracy.</p>
        </div>
        """, unsafe_allow_html=True
    )

with cap_col2:
    st.markdown(
        """
        <div class="card" style="text-align:center;">
            <div style="font-size: 2rem; margin-bottom: 10px;">💰</div>
            <h4 style="margin-bottom: 8px;">Affordability Engine</h4>
            <p style="font-size: 0.9rem; color: #64748b;">Calculate the exact maximum EMI a user can safely afford.</p>
        </div>
        """, unsafe_allow_html=True
    )

with cap_col3:
    st.markdown(
        """
        <div class="card" style="text-align:center;">
            <div style="font-size: 2rem; margin-bottom: 10px;">📊</div>
            <h4 style="margin-bottom: 8px;">Interactive Insights</h4>
            <p style="font-size: 0.9rem; color: #64748b;">Visualizing 22+ financial features for deeper analysis.</p>
        </div>
        """, unsafe_allow_html=True
    )

# Dataset & Tech Stack
st.markdown("### 🧬 Data & Technology")
st.markdown(
    """
    <div style="background-color: #f8fafc; padding: 20px; border-radius: 12px; border: 1px solid #e2e8f0;">
        <ul style="color: #475569; line-height: 1.8;">
            <li><b>Dataset Scale:</b> Trained on a comprehensive dataset of <b>400,000+ financial records</b>.</li>
            <li><b>Feature Engineering:</b> Utilizes 22 key features including <i>Demographics, Employment History, Credit Utilization, and DTI Ratios</i>.</li>
            <li><b>Model Architecture:</b> Hybrid approach combining <b>Classification</b> (XGBoost) for decisioning and <b>Regression</b> (Random Forest) for quantitative limits.</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("<br>", unsafe_allow_html=True)
st.success("👉 Please use the sidebar to navigate to the **EMI Calculator** for a live demonstration.")
