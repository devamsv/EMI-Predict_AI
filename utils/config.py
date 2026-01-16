import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

# Model Paths
CLASSIFIER_PATH = MODELS_DIR / "emi_eligibility_model.pkl"
REGRESSOR_PATH = MODELS_DIR / "max_emi_model.pkl"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.pkl" # Added for safety, though may be built-in

# Data Paths
DATASET_PATH = DATA_DIR / "raw" / "emi_prediction_dataset.csv"

# App Constants
APP_TITLE = "EMIPredict AI"
APP_ICON = "💎"
THEME_COLOR = "#4fabfe"

# Business Rules
HIGH_CREDIT_SCORE_THRESHOLD = 701  # Boosts eligibility
LOW_CREDIT_SCORE_THRESHOLD = 700   # Auto-reject threshold
DTI_HIGH_RISK_THRESHOLD = 80       # Debt-to-Income ratio limit

# Valid training features expected by the model (Order is strictly enforced)
TRAINING_FEATURES = [
    'age', 'gender', 'marital_status', 'education', 'monthly_salary',
    'employment_type', 'years_of_employment', 'company_type', 'house_type',
    'monthly_rent', 'family_size', 'dependents', 'school_fees',
    'college_fees', 'travel_expenses', 'groceries_utilities',
    'other_monthly_expenses', 'existing_loans', 'current_emi_amount',
    'credit_score', 'bank_balance', 'emergency_fund', 'emi_scenario',
    'requested_amount', 'requested_tenure'
]

# Feature Groups for Preprocessing (Recovered from backup)
NUMERICAL_FEATURES = [
    'age', 'monthly_salary', 'years_of_employment', 'monthly_rent',
    'family_size', 'dependents', 'school_fees', 'college_fees', 
    'travel_expenses', 'groceries_utilities', 'other_monthly_expenses',
    'current_emi_amount', 'credit_score', 'bank_balance', 'emergency_fund',
    'requested_amount', 'requested_tenure'
]

CATEGORICAL_FEATURES = [
    'gender', 'marital_status', 'education', 'employment_type', 
    'company_type', 'house_type', 'existing_loans', 'emi_scenario'
]
