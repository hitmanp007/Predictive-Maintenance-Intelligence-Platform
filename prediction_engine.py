import joblib
import pandas as pd
import numpy as np
import shap
import os

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

def explain_machine_prediction(
    machine_type,
    air_temperature,
    process_temperature,
    rotational_speed,
    torque,
    tool_wear
):
    # Prepare machine input
    machine_data = pd.DataFrame([{
        "Type": machine_type,
        "Air temperature [K]": air_temperature,
        "Process temperature [K]": process_temperature,
        "Rotational speed [rpm]": rotational_speed,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear
    }], columns=FEATURE_COLUMNS)

    # Apply the same preprocessing used by the trained model
    transformed_input = preprocessor.transform(machine_data)

    # Generate SHAP explanation
    shap_explanation = explainer(transformed_input)

    # Extract SHAP values for this machine
    shap_values = shap_explanation.values[0]

    # Get transformed feature names
    feature_names = preprocessor.get_feature_names_out()

    # Prepare feature contributions
    contributions = []

    for name, value in zip(feature_names, shap_values):

        # Clean feature names for readability
        readable_name = (
            str(name)
            .replace("num__", "")
            .replace("cat__", "")
            .replace("remainder__", "")
        )

        if readable_name == "Type_H":
            readable_name = "Machine Type H"
        elif readable_name == "Type_L":
            readable_name = "Machine Type L"
        elif readable_name == "Type_M":
            readable_name = "Machine Type M"

        contributions.append({
            "feature": readable_name,
            "shap_value": float(value),
            "influence": (
                "Increases failure risk"
                if value > 0
                else "Reduces failure risk"
                if value < 0
                else "No contribution"
            )
        })

    # Sort by absolute SHAP contribution
    contributions.sort(
        key=lambda item: abs(item["shap_value"]),
        reverse=True
    )

    return contributions


explanation = explain_machine_prediction(
    machine_type="M",
    air_temperature=298.1,
    process_temperature=308.6,
    rotational_speed=1500,
    torque=40,
    tool_wear=120
)

print("\nMachine Prediction Explanation")
print("--------------------------------")

for item in explanation:
    print(
        item["feature"],
        "| SHAP Value:", round(item["shap_value"], 4),
        "|", item["influence"]
    )
    

def predict_multiple_machines(machine_data):

    # Check required columns
    missing_columns = [
        col for col in FEATURE_COLUMNS
        if col not in machine_data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Arrange columns in the correct order
    machine_data = machine_data[FEATURE_COLUMNS].copy()

    # Validate machine types
    if not machine_data["Type"].isin(["L", "M", "H"]).all():
        raise ValueError("Machine Type must be L, M, or H.")

    # Validate numeric inputs
    numeric_columns = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]"
    ]

    for col in numeric_columns:
        machine_data[col] = pd.to_numeric(
            machine_data[col],
            errors="raise"
        )

    if not np.isfinite(
        machine_data[numeric_columns].to_numpy()
    ).all():
        raise ValueError("Machine parameters must be finite.")

    if (
        (machine_data["Air temperature [K]"] <= 0).any()
        or (machine_data["Process temperature [K]"] <= 0).any()
        or (machine_data["Rotational speed [rpm]"] <= 0).any()
        or (machine_data["Torque [Nm]"] < 0).any()
        or (machine_data["Tool wear [min]"] < 0).any()
    ):
        raise ValueError("Invalid machine operating parameters.")

    # Predict failure probabilities
    probabilities = final_model.predict_proba(
        machine_data
    )[:, 1]

    # Apply saved classification threshold
    predictions = (probabilities >= best_threshold).astype(int)

    # Generate results
    results = machine_data.copy()

    results["Failure Probability (%)"] = (
        probabilities * 100
    ).round(2)

    results["Prediction"] = predictions

    results["Machine Status"] = np.where(
        predictions == 1,
        "Failure Risk Detected",
        "Normal"
    )

    return results

    
sample_machines = pd.DataFrame([
    {
        "Type": "M",
        "Air temperature [K]": 298.1,
        "Process temperature [K]": 308.6,
        "Rotational speed [rpm]": 1500,
        "Torque [Nm]": 40,
        "Tool wear [min]": 120
    },
    {
        "Type": "L",
        "Air temperature [K]": 300.0,
        "Process temperature [K]": 310.0,
        "Rotational speed [rpm]": 1400,
        "Torque [Nm]": 50,
        "Tool wear [min]": 200
    }
])

batch_results = predict_multiple_machines(sample_machines)

print("\nMULTIPLE MACHINE PREDICTIONS")
print(batch_results.to_string(index=False))    

def predict_from_csv(file_path):

    # Read CSV file
    machine_data = pd.read_csv(file_path)

    print("\nCSV file loaded successfully!")
    print("Total machine records:", len(machine_data))

    # Check required columns
    missing_columns = [
        col for col in FEATURE_COLUMNS
        if col not in machine_data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if machine_data.empty:
        raise ValueError("The CSV file contains no records.")

    # Predict all machines
    prediction_results = predict_multiple_machines(
        machine_data[FEATURE_COLUMNS]
    )

    # Preserve original CSV columns
    results = machine_data.copy()

    # Add prediction results
    results["Failure Probability (%)"] = (
        prediction_results["Failure Probability (%)"]
    )

    results["Prediction"] = (
        prediction_results["Prediction"]
    )

    results["Machine Status"] = (
        prediction_results["Machine Status"]
    )

    # Create output directory
    os.makedirs("outputs", exist_ok=True)

    # Save prediction report
    output_path = "outputs/machine_predictions.csv"

    results.to_csv(output_path, index=False)

    print("\nPrediction completed successfully!")
    print("Total machines:", len(results))

    print(
        "Normal machines:",
        (results["Prediction"] == 0).sum()
    )

    print(
        "Machines with failure risk:",
        (results["Prediction"] == 1).sum()
    )

    print("\nOutput saved at:", output_path)

    return results


csv_results = predict_from_csv("independent_test_data.csv")




print("\nFirst 10 Prediction Results:")
print(csv_results.head(10).to_string(index=False))

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    average_precision_score
)


def evaluate_csv_predictions(results):

    if "Machine failure" not in results.columns:
        raise ValueError(
            "Actual Machine failure column is missing."
        )

    y_true = results["Machine failure"]
    y_pred = results["Prediction"]
    y_probability = results["Failure Probability (%)"] / 100

    print("\nPREDICTION PERFORMANCE REPORT")
    print("=" * 40)

    print("\nActual Class Distribution:")
    print(y_true.value_counts())

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=["Normal", "Failure"]
        )
    )

    print(
        "PR-AUC:",
        round(average_precision_score(
            y_true,
            y_probability
        ), 4)
    )
    
evaluate_csv_predictions(csv_results)


# generating the automated failure analysis
def generate_failure_analysis(results, top_n=10):
    """
    Generate an automated summary and rank machines
    by predicted failure probability.
    """

    required_columns = [
        "Failure Probability (%)",
        "Prediction",
        "Machine Status"
    ]

    # Validate input
    missing_columns = [
        col for col in required_columns
        if col not in results.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if results.empty:
        raise ValueError("No machine records available for analysis.")

    analysis_data = results.copy()

    # Ensure prediction values are numeric
    analysis_data["Prediction"] = pd.to_numeric(
        analysis_data["Prediction"], errors="raise"
    )

    analysis_data["Failure Probability (%)"] = pd.to_numeric(
        analysis_data["Failure Probability (%)"], errors="raise"
    )

    if not analysis_data["Prediction"].isin([0, 1]).all():
        raise ValueError("Prediction must contain only 0 or 1.")

    if not analysis_data["Failure Probability (%)"].between(0, 100).all():
        raise ValueError("Failure probabilities must be between 0 and 100.")

    # Overall machine statistics
    total_machines = len(analysis_data)

    failure_risk_count = int(
        (analysis_data["Prediction"] == 1).sum()
    )

    normal_count = int(
        (analysis_data["Prediction"] == 0).sum()
    )

    failure_risk_percentage = (
        failure_risk_count / total_machines
    ) * 100

    # Create summary
    summary = {
        "Total Machines Analysed": total_machines,
        "Machines With Failure Risk": failure_risk_count,
        "Normal Machines": normal_count,
        "Failure Risk Percentage": round(
            failure_risk_percentage, 2
        )
    }

    summary_df = pd.DataFrame(
        list(summary.items()),
        columns=["Metric", "Value"]
    )

    # Rank machines by predicted failure probability
    analysis_data.insert(
        0,
        "Machine Record",
        range(1, total_machines + 1)
    )

    top_risk_machines = (
        analysis_data[
            analysis_data["Prediction"] == 1
        ]
        .sort_values(
            by="Failure Probability (%)",
            ascending=False
        )
        .head(top_n)
        .reset_index(drop=True)
    )

    # Save reports
    os.makedirs("outputs", exist_ok=True)

    summary_df.to_csv(
        "outputs/failure_analysis_summary.csv",
        index=False
    )

    top_risk_machines.to_csv(
        "outputs/top_risk_machines.csv",
        index=False
    )

    print("\n========== FAILURE ANALYSIS REPORT ==========")
    print(f"Total Machines Analysed: {total_machines}")
    print(f"Machines With Failure Risk: {failure_risk_count}")
    print(f"Normal Machines: {normal_count}")
    print(f"Failure Risk: {failure_risk_percentage:.2f}%")

    print(f"\n========== TOP {top_n} HIGH-RISK MACHINES ==========")

    if top_risk_machines.empty:
        print("No machines were flagged for failure risk.")
    else:
        print(
            top_risk_machines[
                ["Machine Record", "Failure Probability (%)", "Machine Status"]
            ].to_string(index=False)
        )

    print("\nReports saved successfully:")
    print("- outputs/failure_analysis_summary.csv")
    print("- outputs/top_risk_machines.csv")

    return summary_df, top_risk_machines
    
summary_df, top_risk_machines = generate_failure_analysis(csv_results)
    