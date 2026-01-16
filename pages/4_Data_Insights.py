import streamlit as st
import pandas as pd
import plotly.express as px
from utils.config import DATASET_PATH
from utils.styles import apply_styling

st.set_page_config(page_title="Data Insights - EMIPredict AI", page_icon="📈", layout="wide")
apply_styling()

st.markdown("## 📈 Data Insights & Patterns")
st.markdown("Discover the hidden patterns in financial data that drive our AI predictions.")

@st.cache_data
def load_data():
    try:
        # Load sample for performance
        df = pd.read_csv(DATASET_PATH, nrows=10000)
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

df = load_data()

if df is not None:
    # 1. Dataset Overview
    with st.expander("📂 View Raw Dataset Sample (First 100 Rows)", expanded=False):
        st.dataframe(df.head(100), use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    # 2. Credit Score vs Eligibility (Assumption: 'credit_score' and 'eligibility' columns exist)
    cols = df.columns.astype(str).str.lower()
    
    # Map common column names
    credit_col = next((c for c in df.columns if 'credit' in c.lower()), None)
    eligibility_col = next((c for c in df.columns if 'eligib' in c.lower() or 'target' in c.lower() or 'status' in c.lower()), None)
    salary_col = next((c for c in df.columns if 'salary' in c.lower() or 'income' in c.lower()), None)
    expense_col = next((c for c in df.columns if 'expense' in c.lower()), None)

    if credit_col and eligibility_col:
        with col1:
            st.markdown("### 💳 Credit Score Impact")
            fig = px.box(df, x=eligibility_col, y=credit_col, color=eligibility_col, 
                         title="Credit Score Distribution by Eligibility",
                         color_discrete_map={'Eligible':'#16a34a', 'High_Risk':'#dc2626'})
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={'color': "#1f2937"})
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"Higher {credit_col} generally correlates with better eligibility.")

    if salary_col and expense_col:
        with col2:
            st.markdown("### 💰 Income vs Expenses")
            fig2 = px.scatter(df, x=salary_col, y=expense_col, color=eligibility_col,
                              title="Income vs Expenses Cluster", opacity=0.6,
                              color_discrete_map={'Eligible':'#16a34a', 'High_Risk':'#dc2626'})
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={'color': "#1f2937"})
            st.plotly_chart(fig2, use_container_width=True)
            st.caption("Clusters show safe zones for debt-to-income ratios.")

    # 3. Distribution Analysis
    st.markdown("### 📊 Distribution of Key Features")
    dist_col = st.selectbox("Select Feature to Analyze", [c for c in df.columns if df[c].dtype in ['int64', 'float64']])
    
    fig3 = px.histogram(df, x=dist_col, color=eligibility_col if eligibility_col else None, 
                        marginal="violin", title=f"Distribution of {dist_col}", barmode="overlay")
    fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={'color': "#1f2937"})
    st.plotly_chart(fig3, use_container_width=True)

else:
    st.warning("Data could not be loaded. Please ensure 'data/raw/emi_prediction_dataset.csv' exists.")
