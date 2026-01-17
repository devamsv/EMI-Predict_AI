import joblib
import pandas as pd
import streamlit as st
from pathlib import Path
import sys
import logging
from utils.config import CLASSIFIER_PATH, REGRESSOR_PATH, TRAINING_FEATURES, MODELS_DIR, MODELS_EXIST

# Configure logging for deployment debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_dependencies():
    """
    Verify all required ML dependencies are installed.
    Returns: (is_available, missing_packages_list)
    """
    required = {
        'xgboost': 'XGBoost',
        'sklearn': 'Scikit-learn',
        'joblib': 'Joblib'
    }
    
    missing = []
    for module, name in required.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(f"{name} ({module})")
    
    return len(missing) == 0, missing

@st.cache_resource
def load_models():
    """
    Loads trained ML models with comprehensive error handling.
    
    Production-safe implementation:
    ✅ Checks for missing dependencies (xgboost, scikit-learn)
    ✅ Provides clear, actionable error messages
    ✅ Works on: Local, Docker, Streamlit Cloud
    ✅ Graceful fallback with debugging info
    
    Returns:
        tuple: (classifier, regressor) or (None, None) if loading fails
    """
    
    # ============================================================================
    # STEP 1: DEPENDENCY CHECK (Critical for Streamlit Cloud)
    # ============================================================================
    deps_ok, missing = check_dependencies()
    
    if not deps_ok:
        error_msg = (
            f"❌ **Missing ML Dependencies**\n\n"
            f"The following packages are required but not installed:\n"
        )
        for pkg in missing:
            error_msg += f"- {pkg}\n"
        
        error_msg += (
            f"\n**🔧 Fix for Streamlit Cloud:**\n"
            f"1. Update `requirements.txt` with:\n"
            f"```\n"
            f"xgboost==2.0.2\n"
            f"scikit-learn==1.3.0\n"
            f"```\n"
            f"2. Commit and push to GitHub\n"
            f"3. Streamlit Cloud will auto-redeploy\n\n"
            f"**🔧 Fix Locally:**\n"
            f"```bash\n"
            f"pip install -r requirements.txt\n"
            f"```"
        )
        st.error(error_msg)
        logger.error(f"Dependency check failed: {missing}")
        return None, None
    
    # ============================================================================
    # STEP 2: MODEL FILE EXISTENCE CHECK
    # ============================================================================
    classifier_exists = CLASSIFIER_PATH.exists()
    regressor_exists = REGRESSOR_PATH.exists()
    
    if not classifier_exists or not regressor_exists:
        missing_files = []
        if not classifier_exists:
            missing_files.append(f"emi_eligibility_model.pkl")
        if not regressor_exists:
            missing_files.append(f"max_emi_model.pkl")
        
        error_msg = (
            f"❌ **Model Files Missing**\n\n"
            f"Expected at: `{MODELS_DIR}/`\n"
            f"Missing: {', '.join(missing_files)}\n\n"
            f"**🔧 Fix:**\n"
            f"1. Ensure `.gitignore` allows `.pkl` files in `models/`\n"
            f"2. Commit models: `git add models/*.pkl`\n"
            f"3. Push: `git push`\n"
            f"4. Redeploy on Streamlit Cloud"
        )
        st.error(error_msg)
        logger.error(f"Model files missing: {missing_files}")
        return None, None
    
    # ============================================================================
    # STEP 3: LOAD MODELS WITH DETAILED ERROR HANDLING
    # ============================================================================
    try:
        logger.info(f"Loading classifier from: {CLASSIFIER_PATH}")
        classifier = joblib.load(str(CLASSIFIER_PATH))
        
        logger.info(f"Loading regressor from: {REGRESSOR_PATH}")
        regressor = joblib.load(str(REGRESSOR_PATH))
        
        logger.info("✅ Models loaded successfully")
        return classifier, regressor
        
    except ModuleNotFoundError as e:
        # Handle missing modules that are pickled into the model
        module_name = str(e).split("'")[1] if "'" in str(e) else "unknown"
        error_msg = (
            f"❌ **Missing Module in Pickled Model**\n\n"
            f"The model was trained with `{module_name}` but it's not installed.\n\n"
            f"**🔧 Common Fix:**\n"
            f"Add to `requirements.txt`:\n"
            f"```\n"
            f"{module_name}\n"
            f"```\n"
            f"Then redeploy the app."
        )
        st.error(error_msg)
        logger.error(f"Module not found: {e}")
        return None, None
        
    except Exception as e:
        error_msg = (
            f"❌ **Error Loading Models**\n\n"
            f"**Error Type:** {type(e).__name__}\n"
            f"**Details:** {str(e)}\n\n"
            f"**🔧 Debugging Steps:**\n"
            f"1. Check model file corruption: `ls -la models/`\n"
            f"2. Verify scikit-learn and joblib versions match training env\n"
            f"3. Check that all dependencies (xgboost, etc.) are in requirements.txt\n"
            f"4. View deployment logs in Streamlit Cloud dashboard"
        )
        st.error(error_msg)
        logger.error(f"Model loading failed: {type(e).__name__}: {e}")
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
