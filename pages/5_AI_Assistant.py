import streamlit as st
import time
from utils.styles import apply_styling

st.set_page_config(page_title="AI Assistant - EMIPredict AI", page_icon="🤖", layout="wide")
apply_styling()

st.markdown("## 🤖 Project AI Assistant")
st.markdown("Your personal guide to understanding Financial Risk Assessment.")

# Predefined Q&A Knowledge Base
qa_knowledge_base = {
    "What is EMI Eligibility?": 
        "EMI Eligibility is a measure of a borrower's ability to repay a loan without financial stress. "
        "It considers factors like income, existing debts, credit score, and job stability.",
        
    "Why EMI calculator may fail?":
        "The EMI calculator may fail if inputs are invalid (e.g., zero income, negative values) or if the backend models could not process specific data patterns. "
        "We have implemented strict validation to ensure all inputs are realistic before processing.",

    "How does credit score affect eligibility?":
        "Credit score is a critical factor. Scores above 740 significantly boost your eligibility chances, potentially overriding other minor risk factors. "
        "Scores below 600 generally lead to automatic rejection or high-risk classification.",

    "What is safe EMI?":
        "Safe EMI is the maximum monthly amount you can afford to pay towards a loan without compromising your essential living expenses and existing financial obligations.",

    "How is max EMI calculated?": 
        "This is determined by a **Random Forest Regression** model. It predicts the highest monthly installment "
        "you can pay while maintaining a healthy financial buffer for living expenses and emergencies.",
        
    "Why expenses matter?":
        "Monthly expenses directly impact your disposable income. Higher expenses mean less money available for EMI payments, increasing your Debt-to-Income (DTI) ratio and potential risk.",

    "What models are used?": 
        "We use a hybrid approach: \n"
        "• **XGBoost Classifier** for Eligibility Status (Eligible/Not Eligible/High Risk). \n"
        "• **Random Forest Regressor** for predicting the numeric Safe EMI value.",
        
    "How accurate is this system?": 
        "Our system has been validated on a testing dataset with a **94% Accuracy** for eligibility classification "
        "and an **R² Score of 0.89** for EMI estimation, making it highly reliable for preliminary assessments.",

    "What causes high-risk status?": 
        "A 'High Risk' status often results from a combination of factors: Low Credit Score (<650), "
        "High Debt-to-Income Ratio (>50%), or unstable employment history. Our model flags these "
        "patterns to prevent potential default.",
    
    "Can eligibility be improved?": 
        "Yes! You can improve eligibility by: \n"
        "• Reducing existing EMIs (Foreclosing small loans). \n"
        "• Increasing the loan tenure (This reduces monthly EMI). \n"
        "• Improving your Credit Score (Timely payments). \n"
        "• Adding a co-applicant to combine incomes.",
        
    "Is this system used in real banks?": 
        "This system represents a production-grade prototype similar to what Fintechs and Neo-banks use "
        "for instant loan approvals (digital underwriting). It automates the 'Credit Manager' role using AI.",
}

# Chat Logic
def get_bot_response(user_input):
    user_input = user_input.lower()
    
    # Direct Key Matching
    for question, answer in qa_knowledge_base.items():
        if user_input in question.lower():
            return answer
    
    # Keyword Matching
    if "fail" in user_input or "error" in user_input:
        return qa_knowledge_base["Why EMI calculator may fail?"]
    elif "eligible" in user_input:
        return qa_knowledge_base["What is EMI Eligibility?"]
    elif "score" in user_input:
        return qa_knowledge_base["How does credit score affect eligibility?"]
    elif "safe" in user_input:
         return qa_knowledge_base["What is safe EMI?"]
    elif "expenses" in user_input:
         return qa_knowledge_base["Why expenses matter?"]
    elif "risk" in user_input:
        return qa_knowledge_base["What causes high-risk status?"]
    elif "calculate" in user_input or "max" in user_input:
        return qa_knowledge_base["How is max EMI calculated?"]
    elif "accuracy" in user_input:
        return qa_knowledge_base["How accurate is this system?"]
    elif "improve" in user_input:
        return qa_knowledge_base["Can eligibility be improved?"]
    elif "work" in user_input or "models" in user_input:
        return qa_knowledge_base["What models are used?"]
    elif "hello" in user_input or "hi" in user_input:
        return "Hello! I am your EMIPredict Assistant. You can ask me about eligibility, risk factors, or how our AI models work."
    else:
        return ("I assume you're asking about the project. Here is something relevant: " + 
                qa_knowledge_base["What models are used?"] + 
                "\n\n*Try asking: 'How accurate is this system?' or 'Why am I high risk?'*")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi! Ask me any question listed in the sidebar or type your own!"}]

# layout
col1, col2 = st.columns([1, 3])

with col1:
    st.markdown("### 📌 Common Questions")
    st.markdown("Click to copy & ask:")
    for q in qa_knowledge_base.keys():
        st.markdown(f"- {q}")

with col2:
    # Display Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # User Input
    if prompt := st.chat_input("Ask about the project logic..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        # Generate Response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            response_text = get_bot_response(prompt)
            
            # Simulate typing
            for chunk in response_text.split():
                full_response += chunk + " "
                time.sleep(0.04)
                message_placeholder.write(full_response + "▌")
            message_placeholder.write(full_response)
            
        st.session_state.messages.append({"role": "assistant", "content": full_response})
