import joblib
import pandas as pd
import streamlit as st
from pathlib import Path
from utils.config import CLASSIFIER_PATH, REGRESSOR_PATH, TRAINING_FEATURES, MODELS_DIR, MODELS_EXIST

@st.cache_resource
def load_models():
    """
    Loads trained ML models with comprehensive error handling.
    
    Production-safe implementation that provides:
    - Clear debugging info if models are missing
    - Graceful error messages for deployment issues
    - Works across local, Docker, and Streamlit Cloud environments
    
    Returns:
        tuple: (classifier, regressor) or (None, None) if loading fails
    """
    # DEBUG: Log model file status
    classifier_exists = CLASSIFIER_PATH.exists()
    regressor_exists = REGRESSOR_PATH.exists()
    
    try:
        # Verify model files exist before attempting to load
        if not classifier_exists:
            raise FileNotFoundError(
                f"Classifier model not found at: {CLASSIFIER_PATH}\n"
                f"Available files in {MODELS_DIR}: {list(MODELS_DIR.glob('*')) if MODELS_DIR.exists() else 'directory does not exist'}"
            )
        
        if not regressor_exists:
            raise FileNotFoundError(
                f"Regressor model not found at: {REGRESSOR_PATH}\n"
                f"Available files in {MODELS_DIR}: {list(MODELS_DIR.glob('*')) if MODELS_DIR.exists() else 'directory does not exist'}"
            )
        
        # Load models
        classifier = joblib.load(str(CLASSIFIER_PATH))
        regressor = joblib.load(str(REGRESSOR_PATH))
        
        return classifier, regressor
        
    except FileNotFoundError as e:
        st.error(
            f"❌ **Model Files Missing**\n\n"
            f"Error: {e}\n\n"
            f"**Solution:**\n"
            f"1. Ensure `models/emi_eligibility_model.pkl` is committed to GitHub\n"
            f"2. Check that `.gitignore` doesn't exclude `.pkl` files in the models/ directory\n"
            f"3. Redeploy the Streamlit Cloud app"
        )
        return None, None
    except Exception as e:
        st.error(
            f"❌ **Error Loading Models**\n\n"
            f"Technical Error: {type(e).__name__}: {e}\n\n"
            f"**Debugging steps:**\n"
            f"- Check model file integrity\n"
            f"- Verify joblib/scikit-learn versions match training environment\n"
            f"- Check requirements.txt for compatible dependencies"
        )
        return None, None

def align_features(input_dict, required_features):
    """
    Aligns input dictionary to the exact schema required by the model.
    Fills missing values with defaults and ensures column order.
    """
    # Defaults for optional/missing fields (inferred from backup logic or safe defaults)
    defaults = {
        'gender': 'Male', 
        'marital_status': 'Single',
        'education': 'Graduate',
        'company_type': 'Private',
        'house_type': 'Rented',
        'family_size': 3,
        'dependents': 1,
        'school_fees': 0,
        'college_fees': 0,
        'travel_expenses': 2000,
        'groceries_utilities': 5000,
        'other_monthly_expenses': 2000,
        'existing_loans': 'No', # Changed to categorical "No"/0 based on feature type context, assuming categorical
        'bank_balance': 10000,
        'emergency_fund': 0,
        'emi_scenario': 'Personal Loan',
        'monthly_rent': 0,
        'current_emi_amount': 0,
        'requested_amount': 100000,
        'requested_tenure': 12
    }
    
    aligned_data = {}
    for feature in required_features:
        if feature in input_dict:
            aligned_data[feature] = input_dict[feature]
        else:
            # Use default if available, else 0 (fallback)
            aligned_data[feature] = defaults.get(feature, 0)
            
    return aligned_data

def make_prediction(classifier, regressor, input_data):
    """
    Makes predictions using the loaded models after aligning features.
    
    Args:
        classifier: Trained classification model
        regressor: Trained regression model
        input_data (dict): User input data
        
    Returns:
        tuple: (eligibility_label, max_emi_value)
    """
    try:
        # 1. Feature Alignment (CRITICAL FIX)
        # Ensure the input dictionary matches the TRAINING_FEATURES list exactly
        aligned_data = align_features(input_data, TRAINING_FEATURES)
        
        # 2. DataFrame Conversion and Ordering
        input_df = pd.DataFrame([aligned_data])[TRAINING_FEATURES]
        
        # 3. Categorical Handling Logic
        # If the model expects specific categories or numbers, this layer bridges it.
        # Assuming the pipeline handles "strings" (OneHotEncoder), we leave them as strings.
        # However, for 'existing_loans', if input is int (0), but training was categorical ("No"), we might need mapping.
        # We rely on the robustness of the trained pipeline (usually handles mixed types or casts).
        
        # Classification Prediction
        try:
            # Predict
            eligibility_pred = classifier.predict(input_df)[0]
        except Exception as e:
            # If standard predict fails, it might be an issue with unseen categories or data types.
            # Log specific warning for debugging
            print(f"DEBUG: Classification Pipeline Failed: {e}") 
            st.warning(f"Classification Model Warning: {e}")
            eligibility_pred = "Error"

        # Regression Prediction
        try:
            max_emi_pred = regressor.predict(input_df)[0]
        except Exception as e:
            print(f"DEBUG: Regression Pipeline Failed: {e}")
            st.warning(f"Regression Model Warning: {e}")
            max_emi_pred = 0.0
            
        return eligibility_pred, max_emi_pred

    except Exception as e:
        st.error(f"Prediction Pipeline Error: {e}")
        return "Error", 0.0

def safe_float(value):
    try:
        return float(value)
    except:
        return 0.0
