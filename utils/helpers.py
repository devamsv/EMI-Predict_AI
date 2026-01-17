"""
Model loading and prediction utilities for EMI Predict AI.
Handles ML model inference with production-grade error handling.
"""

import logging
from pathlib import Path

# Safe imports with fallback handling
try:
    import joblib
except ImportError:
    raise ImportError("joblib is required. Install with: pip install joblib")

try:
    import pandas as pd
except ImportError:
    raise ImportError("pandas is required. Install with: pip install pandas")

try:
    import streamlit as st
except ImportError:
    raise ImportError("streamlit is required. Install with: pip install streamlit")

try:
    from utils.config import (
        CLASSIFIER_PATH,
        REGRESSOR_PATH,
        TRAINING_FEATURES,
        MODELS_DIR,
    )
except ImportError as e:
    raise ImportError(f"Failed to import from utils.config: {e}")


# Configure logging for deployment debugging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_dependencies():
    """
    Verify all required ML dependencies are installed.
    
    Returns:
        tuple: (is_available: bool, missing_packages: list)
    """
    required = {
        'xgboost': 'XGBoost (ML model inference)',
        'sklearn': 'Scikit-learn (ML pipeline)',
        'joblib': 'Joblib (Model serialization)',
    }
    
    missing = []
    for module, description in required.items():
        try:
            __import__(module)
            logger.info(f"✅ {description} is available")
        except ImportError:
            logger.warning(f"❌ {description} is NOT available")
            missing.append(f"{description.split()[0]} ({module})")
    
    return len(missing) == 0, missing


@st.cache_resource
def load_models():
    """
    Loads trained ML models with comprehensive production-grade error handling.
    
    This function:
    ✅ Verifies all ML dependencies are installed
    ✅ Checks model files exist before loading
    ✅ Provides detailed, actionable error messages
    ✅ Works on: Local, Docker, Streamlit Cloud
    ✅ Logs all operations for debugging
    
    Returns:
        tuple: (classifier, regressor) or (None, None) if loading fails
    """
    
    logger.info("=" * 80)
    logger.info("MODEL LOADING INITIATED")
    logger.info("=" * 80)
    
    # ============================================================================
    # STEP 1: DEPENDENCY CHECK (Critical for Streamlit Cloud)
    # ============================================================================
    logger.info("STEP 1: Checking ML dependencies...")
    deps_ok, missing = check_dependencies()
    
    if not deps_ok:
        error_msg = (
            "❌ **Missing ML Dependencies**\n\n"
            "The following packages are required but not installed:\n"
        )
        for pkg in missing:
            error_msg += f"- {pkg}\n"
        
        error_msg += (
            "\n**🔧 How to Fix on Streamlit Cloud:**\n"
            "1. Check your `requirements.txt` includes:\n"
            "   - `xgboost>=2.0.0`\n"
            "   - `scikit-learn>=1.3.0`\n"
            "   - `joblib>=1.3.0`\n"
            "2. Create a `runtime.txt` with: `python-3.11.7`\n"
            "3. Commit and push to GitHub\n"
            "4. Streamlit Cloud will auto-redeploy with dependencies\n\n"
            "**🔧 How to Fix Locally:**\n"
            "```bash\n"
            "pip install -r requirements.txt\n"
            "```"
        )
        st.error(error_msg)
        logger.error(f"DEPENDENCY CHECK FAILED: {missing}")
        return None, None
    
    logger.info("✅ All dependencies are available")
    
    # ============================================================================
    # STEP 2: MODEL FILE EXISTENCE CHECK
    # ============================================================================
    logger.info("STEP 2: Verifying model files exist...")
    classifier_exists = CLASSIFIER_PATH.exists()
    regressor_exists = REGRESSOR_PATH.exists()
    
    logger.info(f"  Classifier path: {CLASSIFIER_PATH}")
    logger.info(f"  Classifier exists: {classifier_exists}")
    logger.info(f"  Regressor path: {REGRESSOR_PATH}")
    logger.info(f"  Regressor exists: {regressor_exists}")
    
    if not classifier_exists or not regressor_exists:
        missing_files = []
        if not classifier_exists:
            missing_files.append("emi_eligibility_model.pkl")
        if not regressor_exists:
            missing_files.append("max_emi_model.pkl")
        
        error_msg = (
            "❌ **Model Files Missing**\n\n"
            f"Expected location: `{MODELS_DIR}/`\n"
            f"Missing files: {', '.join(missing_files)}\n\n"
            "**🔧 How to Fix:**\n"
            "1. Ensure models are committed to GitHub:\n"
            "   ```bash\n"
            "   git add models/*.pkl\n"
            "   ```\n"
            "2. Check `.gitignore` doesn't exclude `.pkl` files in `models/`\n"
            "3. Push changes:\n"
            "   ```bash\n"
            "   git push\n"
            "   ```\n"
            "4. Redeploy on Streamlit Cloud"
        )
        st.error(error_msg)
        logger.error(f"MODEL FILES MISSING: {missing_files}")
        return None, None
    
    logger.info("✅ All model files are present")
    
    # ============================================================================
    # STEP 3: LOAD MODELS WITH DETAILED ERROR HANDLING
    # ============================================================================
    logger.info("STEP 3: Loading models from disk...")
    
    try:
        logger.info(f"Loading classifier: {CLASSIFIER_PATH}")
        classifier = joblib.load(str(CLASSIFIER_PATH))
        logger.info("✅ Classifier loaded successfully")
        
        logger.info(f"Loading regressor: {REGRESSOR_PATH}")
        regressor = joblib.load(str(REGRESSOR_PATH))
        logger.info("✅ Regressor loaded successfully")
        
        logger.info("=" * 80)
        logger.info("✅ ALL MODELS LOADED SUCCESSFULLY")
        logger.info("=" * 80)
        return classifier, regressor
        
    except ModuleNotFoundError as e:
        # Handle missing modules that are pickled into the model
        module_name = str(e).split("'")[1] if "'" in str(e) else "unknown"
        error_msg = (
            "❌ **Missing Module in Model**\n\n"
            f"The model requires `{module_name}` but it's not installed.\n\n"
            "**🔧 How to Fix:**\n"
            "1. Add the missing dependency to `requirements.txt`:\n"
            f"   ```\n"
            f"   {module_name}\n"
            f"   ```\n"
            "2. Push to GitHub\n"
            "3. Streamlit Cloud will auto-redeploy"
        )
        st.error(error_msg)
        logger.error(f"MODULE NOT FOUND: {module_name}", exc_info=True)
        return None, None
        
    except Exception as e:
        error_msg = (
            "❌ **Error Loading Models**\n\n"
            f"**Error Type:** `{type(e).__name__}`\n"
            f"**Message:** {str(e)}\n\n"
            "**🔧 Debugging Steps:**\n"
            "1. Verify model files are not corrupted:\n"
            "   ```bash\n"
            "   ls -lh models/\n"
            "   ```\n"
            "2. Check Python and package versions match training environment\n"
            "3. Review requirements.txt for compatibility\n"
            "4. Check Streamlit Cloud logs for more details\n"
            "5. Try recreating models if files are corrupted"
        )
        st.error(error_msg)
        logger.error(f"MODEL LOADING FAILED: {type(e).__name__}: {str(e)}", exc_info=True)
        return None, None


def align_features(input_dict, required_features):
    """
    Aligns input dictionary to the exact schema required by the model.
    Fills missing values with defaults and ensures correct column order.
    
    Args:
        input_dict (dict): User-provided input data
        required_features (list): Features expected by the model
        
    Returns:
        dict: Aligned data matching model schema
    """
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
        'existing_loans': 'No',
        'bank_balance': 10000,
        'emergency_fund': 0,
        'emi_scenario': 'Personal Loan',
        'monthly_rent': 0,
        'current_emi_amount': 0,
        'requested_amount': 100000,
        'requested_tenure': 12,
    }
    
    aligned_data = {}
    for feature in required_features:
        if feature in input_dict:
            aligned_data[feature] = input_dict[feature]
        else:
            aligned_data[feature] = defaults.get(feature, 0)
    
    return aligned_data


def make_prediction(classifier, regressor, input_data):
    """
    Makes eligibility and EMI predictions using loaded models.
    
    Args:
        classifier: Trained classification model
        regressor: Trained regression model
        input_data (dict): User input features
        
    Returns:
        tuple: (eligibility_label, max_emi_value) or ("Error", 0.0) on failure
    """
    try:
        # Align features to model schema
        aligned_data = align_features(input_data, TRAINING_FEATURES)
        
        # Create DataFrame with correct column order
        input_df = pd.DataFrame([aligned_data])[TRAINING_FEATURES]
        
        # Classification prediction (eligibility)
        try:
            eligibility_pred = classifier.predict(input_df)[0]
            logger.info(f"Classification prediction: {eligibility_pred}")
        except Exception as e:
            logger.warning(f"Classification failed: {e}")
            st.warning(f"⚠️ Eligibility prediction error: {e}")
            eligibility_pred = "Error"
        
        # Regression prediction (max EMI)
        try:
            max_emi_pred = regressor.predict(input_df)[0]
            logger.info(f"Regression prediction: {max_emi_pred}")
        except Exception as e:
            logger.warning(f"Regression failed: {e}")
            st.warning(f"⚠️ EMI calculation error: {e}")
            max_emi_pred = 0.0
        
        return eligibility_pred, max_emi_pred
    
    except Exception as e:
        logger.error(f"Prediction pipeline error: {e}", exc_info=True)
        st.error(f"❌ Error during prediction: {e}")
        return "Error", 0.0


def safe_float(value):
    """Convert value to float safely."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0
