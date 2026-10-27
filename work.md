
PHASE 1 — UNDERSTAND THE DATA
    Load the CSV
    Inspect dimensions and column names
    Identify feature types and target
    Check missing values, duplicates and distributions
    Investigate the proportion of failure cases

PHASE 2 — PREPARE THE DATA
    Remove identifiers from predictive features
    Decide how to handle missing or invalid values
    Encode categorical features
    Separate X (features) and y (target)
    Create stratified train, validation and test splits

PHASE 3 — BUILD A BASELINE
    Establish a simple majority-class baseline
    Train logistic regression
    Evaluate precision, recall, F1 and PR-AUC

PHASE 4 — IMPROVE THE MODEL
    Train a decision tree and random forest
    Try gradient boosting or XGBoost
    Tune model settings using training/validation data
    Compare models at an appropriate decision threshold

PHASE 5 — EXPLAIN AND VALIDATE
    Examine false negatives and false positives
    Use confusion matrix and precision-recall curve
    Explain feature contributions with SHAP
    Evaluate once on the untouched test set

PHASE 6 — BUILD THE PLATFORM
    Save the preprocessing pipeline and model
    Create a Streamlit dashboard
    Allow sensor values to be entered or uploaded
    Display predictions, explanations and risk categories
    Record predictions and their timestamps

PHASE 7 — FUTURE IMPROVEMENTS
    Process incoming sensor readings
    Monitor changes in model performance
    Add time-series degradation modeling
    Estimate remaining useful life using NASA data
