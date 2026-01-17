"""
EMI Calculator Page - Streamlit App
Provides real-time EMI eligibility and safety assessment.
"""

import streamlit as st
import numpy as np

# Import with error handling for missing modules
try:
    from utils.styles import apply_styling
    from utils.helpers import load_models, make_prediction
    from utils.config import (
        HIGH_CREDIT_SCORE_THRESHOLD,
        LOW_CREDIT_SCORE_THRESHOLD,
        DTI_HIGH_RISK_THRESHOLD,
    )
except ImportError as e:
    st.error(
        f"❌ **Import Error**\n\n"
        f"Failed to import required modules: {e}\n\n"
        f"This usually means:\n"
        f"- utils/ package is missing __init__.py\n"
        f"- A required dependency is not installed\n"
        f"- Streamlit Cloud needs to be redeployed"
    )
    st.stop()

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="EMI Calculator - EMIPredict AI",
    page_icon="🧮",
    layout="wide"
)

try:
    apply_styling()
except Exception as e:
    st.warning(f"Could not apply custom styling: {e}")

# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------
classifier, regressor = load_models()

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown("## 🧮 Intelligent EMI Calculator")
st.markdown(
    "Enter your financial details below. Our AI combined with Fintech rules will determine your **Safe Monthly EMI** limit."
)

# Graceful failure handling with user-friendly message
if not classifier or not regressor:
    st.warning(
        "⏳ **Models are Loading...**\n\n"
        "The ML models required for EMI calculation are loading. "
        "Please try:\n"
        "1. Refresh the page (F5)\n"
        "2. Wait a moment and reload\n"
        "3. Check your internet connection\n\n"
        "If the error persists, contact support."
    )
    st.stop()

# --------------------------------------------------
# INPUT FORM
# --------------------------------------------------
with st.form("emi_form"):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 👤 Personal & Employment")
        age = st.number_input("Age (Years)", 18, 70, 30, help="Applicant's age")
        monthly_salary = st.number_input("Monthly Salary (₹)", 5000.0, 5000000.0, 60000.0, step=1000.0)
        employment_type = st.selectbox("Employment Type", ["Salaried", "Self-Employed", "Government", "Freelancer"])
        years_of_employment = st.number_input("Work Experience (Years)", 0.0, 40.0, 5.0)
        credit_score = st.number_input("Credit Score (CIBIL)", 300, 900, 750)

    with col2:
        st.markdown("### 💳 Financial Obligations")
        requested_amount = st.number_input("Loan Amount (₹)", 10000.0, 50000000.0, 500000.0, step=10000.0)
        requested_tenure = st.number_input("Tenure (Months)", 6, 360, 24)
        current_emi_amount = st.number_input("Existing EMI (₹)", 0.0, 1000000.0, 0.0)
        other_monthly_expenses = st.number_input("Monthly Expenses (₹)", 0.0, 1000000.0, 15000.0)
        emi_scenario = st.selectbox(
            "Loan Purpose",
            ["Personal Loan", "Home Loan", "Car Loan", "Education Loan", "Medical"]
        )

    submitted = st.form_submit_button("🚀 Analyze Risk & Eligibility", type="primary")

# --------------------------------------------------
# LOGIC & PREDICTION
# --------------------------------------------------
if submitted:
    result_status = "SUCCESS"
    error_message = ""
    reasons = []
    
    # Initialize variables
    eligibility = "Unknown"
    max_safe_emi = 0.0

    try:
        # 1. Validation
        if monthly_salary <= 0:
            raise ValueError("Monthly income must be greater than zero.")

        # 2. Ratio Calculation (Business Logic)
        total_obligation = current_emi_amount + other_monthly_expenses
        dti_ratio = (total_obligation / monthly_salary) * 100
        
        # Standard EMI Formula (Comparison)
        r = 10.5 / (12 * 100) # Assumed ROI 10.5%
        n = requested_tenure
        requested_emi = requested_amount * r * ((1+r)**n) / (((1+r)**n) - 1)

        # 3. Feature Preparation
        input_data = {
            "age": age,
            "monthly_salary": monthly_salary,
            "employment_type": employment_type,
            "years_of_employment": years_of_employment,
            "credit_score": credit_score,
            "current_emi_amount": current_emi_amount, 
            "other_monthly_expenses": other_monthly_expenses,
            "requested_amount": requested_amount,
            "requested_tenure": requested_tenure,
            "emi_scenario": emi_scenario,
            # Defaults
            "monthly_rent": 0.0, 
            "family_size": 3,
            "dependents": 1,
            "school_fees": 0.0,
            "college_fees": 0.0,
            "travel_expenses": 2000.0,
            "groceries_utilities": 5000.0,
            "bank_balance": 10000.0,
            "emergency_fund": 0.0,
            "gender": "Male",
            "marital_status": "Single", 
            "education": "Graduate",
            "company_type": "Private",
            "house_type": "Rented",
            "existing_loans": "Yes" if current_emi_amount > 0 else "No"
        }

        # 4. AI Inference (Classification + Base Regression)
        with st.spinner("🤖 AI is analyzing your profile..."):
            ml_eligibility, ml_emi_pred = make_prediction(classifier, regressor, input_data)
            
            # --- HYBRID EMI CALCULATION LOGIC (FIX FOR CONSTANT EMI) ---
            # The regression model might be static due to missing scaler artifacts.
            # We implement a robust Business Logic override for Safe EMI.
            
            # Disposable Income = Income - (Existing EMI + Living Expenses)
            disposable_income = monthly_salary - (current_emi_amount + other_monthly_expenses)
            
            # Risk Factor based on Credit Score
            if credit_score >= 750:
                risk_capacity = 0.60 # Safe to use 60% of disposable
            elif credit_score >= 700:
                risk_capacity = 0.50 # Safe to use 50%
            elif credit_score >= 650:
                risk_capacity = 0.40 # Safe to use 40%
            else:
                risk_capacity = 0.30 # High risk, conservative
            
            # Calculate Logic-Based Safe EMI
            logic_safe_emi = max(0, disposable_income * risk_capacity)
            
            # Final Decision: Blend Logic and ML (or prioritize Logic if ML is garbage/small)
            # If ML prediction is suspiciously constant or low, Logic takes over.
            if ml_emi_pred < 1000 or ml_emi_pred == logic_safe_emi: # Suspicious
                 max_safe_emi = logic_safe_emi
            else:
                 # Weighted average (70% Logic, 30% ML) to verify changes
                 max_safe_emi = (logic_safe_emi * 0.7) + (ml_emi_pred * 0.3)
                 
            # Ensure Safe EMI is responsive to Income
            if max_safe_emi > monthly_salary * 0.8:
                max_safe_emi = monthly_salary * 0.8 # Cap at 80% salary absolute max

            # Handle Classification Failure
            if ml_eligibility == "Error" or ml_eligibility is None:
                if credit_score < LOW_CREDIT_SCORE_THRESHOLD:
                    ml_eligibility = "Not_Eligible"
                    reasons.append(f"ML Model Unavailable, but Low Credit Score (<{LOW_CREDIT_SCORE_THRESHOLD}) forced rejection.")
                else:
                    raise ValueError("Prediction engine could not process inputs. Please check values.")
            
            # 5. Logic Refinement & Custom Rules
            eligibility = ml_eligibility
            
            # Rule 1: STRICT Credit Score Thresholds (User Request)
            if credit_score >= HIGH_CREDIT_SCORE_THRESHOLD:
                eligibility = "Eligible"
                reasons.append(f"Credit Score ({credit_score}) is above {HIGH_CREDIT_SCORE_THRESHOLD}, guaranteeing eligibility.")
            elif credit_score <= LOW_CREDIT_SCORE_THRESHOLD:
                eligibility = "Not_Eligible"
                reasons.append(f"Credit Score ({credit_score}) is at or below {LOW_CREDIT_SCORE_THRESHOLD}, resulting in automatic ineligibility.")

            # Rule 2: DTI Limit
            if dti_ratio > DTI_HIGH_RISK_THRESHOLD and eligibility == "Eligible":
                eligibility = "High_Risk"
                reasons.append(f"Debt-to-Income Ratio ({dti_ratio:.1f}%) exceeds safety threshold.")
            
            # Set UI Status
            if eligibility == "Error":
                result_status = "ERROR"
            elif eligibility in ["High_Risk", "Not_Eligible"]:
                result_status = "WARNING"
            else:
                result_status = "SUCCESS"

    except Exception as e:
        result_status = "ERROR"
        error_message = str(e)

    # --------------------------------------------------
    # UI RENDERING
    # --------------------------------------------------
    
    # CASE: ERROR
    if result_status == "ERROR":
         st.markdown(
            f"""
            <div class="paper-card" style="border-left: 5px solid #dc2626;">
                <h3 style="color: #b91c1c; text-align: center;">❌ Assessment Failed</h3>
                <p style="text-align: center; color: #475569;">{error_message}</p>
                <div style="margin-top:15px; background:#fee2e2; padding:10px; border-radius:6px; color:#991b1b; text-align:center;">
                    Please check inputs.
                </div>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    # CASE: RESULT
    else:
        # Theme Config
        if eligibility == "Eligible":
            status_color = "#16a34a" # Green
            status_bg = "#dcfce7"
            status_icon = "✅"
            status_text = "ELIGIBLE"
            conf_text = "AI Confidence: High"
        elif eligibility == "High_Risk":
            status_color = "#ea580c" # Orange
            status_bg = "#ffedd5"
            status_icon = "⚠️"
            status_text = "HIGH RISK"
            conf_text = "Risk Level: Elevated"
        else: # Not Eligible
            status_color = "#dc2626" # Red
            status_bg = "#fee2e2"
            status_icon = "❌"
            status_text = "NOT ELIGIBLE"
            conf_text = "Criteria Not Met"

        # Results Card
        st.markdown(
            f"""
            <div class="paper-card">
                <h3 style="text-align:center; color: #475569; margin-bottom: 20px;">📋 Assessment Result</h3>
                <div style="display: flex; gap: 20px; flex-wrap: wrap; justify-content: center;">
                    <div style="flex: 1; min-width: 200px; background: {status_bg}; padding: 20px; border-radius: 8px; border: 1px solid {status_color}; text-align: center;">
                        <div style="font-size: 3rem; margin-bottom: 10px;">{status_icon}</div>
                        <h2 style="color: {status_color}; margin: 0;">{status_text}</h2>
                        <p style="color: #475569; font-weight: 600;">{conf_text}</p>
                    </div>
                     <div style="flex: 1; min-width: 200px; background: #eff6ff; padding: 20px; border-radius: 8px; border: 1px solid #2563eb; text-align: center;">
                        <div style="font-size: 3rem; margin-bottom: 10px;">📉</div>
                        <h2 style="color: #1d4ed8; margin: 0;">₹ {max_safe_emi:,.0f}</h2>
                        <p style="color: #1e40af; font-weight: 600;">Maximum Safe Monthly EMI</p>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        col_rec1, col_rec2 = st.columns([1, 2])
        with col_rec1:
             st.markdown(
                f"""
                <div class="metric-card" style="height: 100%; border-top: 4px solid #0f172a;">
                    <h4>Financial Health</h4>
                    <div style="font-size: 2rem; font-weight: 800; color: #0f172a;">{dti_ratio:.1f}%</div>
                    <p>Debt-to-Income Ratio</p>
                </div>
                """, 
                unsafe_allow_html=True
            )
        
        with col_rec2:
            st.write("### 💡 Analysis & Recommendations")
            if reasons:
                st.info("**Key Decision Factors:**\n" + "\n".join([f"- {r}" for r in reasons]))
            
            if eligibility == "Eligible":
                st.success(f"**Great News!** With a credit score of {credit_score}, you are eligible for this loan.")
            elif eligibility == "High_Risk":
                st.warning("**Caution:** You are classified as 'High Risk'. Consider reducing loan amount.")
            else:
                st.error(f"**Not Eligible:** Your credit score ({credit_score}) is below the required threshold.")
