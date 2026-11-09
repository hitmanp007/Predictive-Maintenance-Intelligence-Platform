import joblib
import pandas as pd
import numpy as np
import shap

# Load the trained XGBoost pipeline
final_model = joblib.load(
    "models/xgboost_predictive_maintenance.pkl"
)

# Load the optimized classification threshold
best_threshold = joblib.load(
    "models/xgboost_threshold.pkl"
)

print("Model loaded successfully!")
print("Threshold:", best_threshold)
print("Pipeline steps:", list(final_model.named_steps.keys()))


# Expected input features
FEATURE_COLUMNS = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]


def predict_machine(
    machine_type,
    air_temperature,
    process_temperature,
    rotational_speed,
    torque,
    tool_wear
):
    # Validate machine type
    if machine_type not in ["L", "M", "H"]:
        raise ValueError("Machine type must be L, M, or H.")

    # Validate numeric inputs
    numeric_values = [
        air_temperature,
        process_temperature,
        rotational_speed,
        torque,
        tool_wear
    ]

    if not all(np.isfinite(value) for value in numeric_values):
        raise ValueError("All machine parameters must be finite numbers.")

    if air_temperature <= 0 or process_temperature <= 0:
        raise ValueError("Temperatures must be greater than zero.")

    if rotational_speed <= 0 or torque < 0 or tool_wear < 0:
        raise ValueError("Invalid machine operating parameters.")

    # Create input DataFrame
    machine_data = pd.DataFrame([{
        "Type": machine_type,
        "Air temperature [K]": air_temperature,
        "Process temperature [K]": process_temperature,
        "Rotational speed [rpm]": rotational_speed,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear
    }], columns=FEATURE_COLUMNS)

    # Predict failure probability
    failure_probability = final_model.predict_proba(
        machine_data
    )[0][1]

    # Apply optimized threshold
    prediction = int(failure_probability >= best_threshold)

    # Prepare result
    result = {
        "failure_probability": round(float(failure_probability), 4),
        "failure_percentage": round(float(failure_probability * 100), 2),
        "prediction": prediction,
        "status": "Failure Risk Detected" if prediction == 1 else "Normal"
    }

    return result


# Test machine input
test_result = predict_machine(
    machine_type="M",
    air_temperature=298.1,
    process_temperature=308.6,
    rotational_speed=1500,
    torque=40,
    tool_wear=120
)

print("Machine Prediction")
print("-------------------------")
print("Failure Probability:", test_result["failure_percentage"], "%")
print("Prediction:", test_result["prediction"])
print("Machine Status:", test_result["status"])


# Extract preprocessing and XGBoost classifier
preprocessor = final_model.named_steps["preprocessing"]
xgb_classifier = final_model.named_steps["classifier"]

# Initialize SHAP TreeExplainer
explainer = shap.TreeExplainer(xgb_classifier)

print("SHAP Explainer initialized successfully!")