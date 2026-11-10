"""
Predictive Maintenance Intelligence Platform
--------------------------------------------
Core prediction, evaluation, failure-analysis, and SHAP functions.

Run directly for a local smoke test:
    python prediction_engine.py

Import from Streamlit without running the test suite:
    from prediction_engine import predict_machine, predict_multiple_machines, ...
"""

from pathlib import Path
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
import shap

from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
)


# ============================================================
# 1. PATHS AND MODEL CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

MODEL_PATH = MODEL_DIR / "xgboost_predictive_maintenance.pkl"
THRESHOLD_PATH = MODEL_DIR / "xgboost_threshold.pkl"

FEATURE_COLUMNS = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

NUMERIC_COLUMNS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

VALID_MACHINE_TYPES = {"L", "M", "H"}


# ============================================================
# 2. LOAD TRAINED MODEL AND THRESHOLD
# ============================================================

def load_model_assets():
    """Load the trained pipeline and its optimized threshold."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    if not THRESHOLD_PATH.exists():
        raise FileNotFoundError(
            f"Classification threshold not found: {THRESHOLD_PATH}"
        )

    model = joblib.load(MODEL_PATH)
    threshold = float(joblib.load(THRESHOLD_PATH))

    if not 0 <= threshold <= 1:
        raise ValueError("Saved classification threshold must be between 0 and 1.")

    required_steps = {"preprocessing", "classifier"}
    if not hasattr(model, "named_steps") or not required_steps.issubset(
        model.named_steps
    ):
        raise ValueError(
            "The saved model must be a pipeline containing "
            "'preprocessing' and 'classifier' steps."
        )

    return model, threshold


final_model, best_threshold = load_model_assets()

# Keep these references available for existing code that imports them.
preprocessor = final_model.named_steps["preprocessing"]
xgb_classifier = final_model.named_steps["classifier"]


@lru_cache(maxsize=1)
def get_shap_explainer():
    """Create the SHAP TreeExplainer once, when first needed."""
    return shap.TreeExplainer(xgb_classifier)


# ============================================================
# 3. INPUT VALIDATION HELPERS
# ============================================================

def _validate_single_machine(
    machine_type,
    air_temperature,
    process_temperature,
    rotational_speed,
    torque,
    tool_wear,
):
    """Validate one machine's input values."""
    if machine_type not in VALID_MACHINE_TYPES:
        raise ValueError("Machine type must be L, M, or H.")

    numeric_values = [
        air_temperature,
        process_temperature,
        rotational_speed,
        torque,
        tool_wear,
    ]

    try:
        numeric_values = [float(value) for value in numeric_values]
    except (TypeError, ValueError):
        raise ValueError("All machine parameters must be numeric.") from None

    if not np.isfinite(numeric_values).all():
        raise ValueError("All machine parameters must be finite numbers.")

    air_temperature, process_temperature, rotational_speed, torque, tool_wear = (
        numeric_values
    )

    if air_temperature <= 0 or process_temperature <= 0:
        raise ValueError("Temperatures must be greater than zero.")

    if rotational_speed <= 0 or torque < 0 or tool_wear < 0:
        raise ValueError("Invalid machine operating parameters.")

    return {
        "Type": machine_type,
        "Air temperature [K]": air_temperature,
        "Process temperature [K]": process_temperature,
        "Rotational speed [rpm]": rotational_speed,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear,
    }


def _validate_machine_dataframe(machine_data):
    """Validate and return a clean dataframe in model feature order."""
    if not isinstance(machine_data, pd.DataFrame):
        raise TypeError("machine_data must be a pandas DataFrame.")

    missing_columns = [
        column for column in FEATURE_COLUMNS
        if column not in machine_data.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if machine_data.empty:
        raise ValueError("The input contains no machine records.")

    clean_data = machine_data[FEATURE_COLUMNS].copy()

    if not clean_data["Type"].isin(VALID_MACHINE_TYPES).all():
        raise ValueError("Machine Type must contain only L, M, or H.")

    for column in NUMERIC_COLUMNS:
        clean_data[column] = pd.to_numeric(
            clean_data[column],
            errors="raise",
        )

    numeric_values = clean_data[NUMERIC_COLUMNS].to_numpy(dtype=float)

    if not np.isfinite(numeric_values).all():
        raise ValueError("Machine parameters must be finite numbers.")

    if (
        (clean_data["Air temperature [K]"] <= 0).any()
        or (clean_data["Process temperature [K]"] <= 0).any()
        or (clean_data["Rotational speed [rpm]"] <= 0).any()
        or (clean_data["Torque [Nm]"] < 0).any()
        or (clean_data["Tool wear [min]"] < 0).any()
    ):
        raise ValueError("Invalid machine operating parameters.")

    return clean_data


# ============================================================
# 4. SINGLE-MACHINE PREDICTION
# ============================================================

def predict_machine(
    machine_type,
    air_temperature,
    process_temperature,
    rotational_speed,
    torque,
    tool_wear,
):
    """Predict failure risk for one machine."""
    machine_record = _validate_single_machine(
        machine_type,
        air_temperature,
        process_temperature,
        rotational_speed,
        torque,
        tool_wear,
    )

    machine_data = pd.DataFrame(
        [machine_record],
        columns=FEATURE_COLUMNS,
    )

    failure_probability = float(
        final_model.predict_proba(machine_data)[0, 1]
    )
    prediction = int(failure_probability >= best_threshold)

    return {
        "failure_probability": round(failure_probability, 4),
        "failure_percentage": round(failure_probability * 100, 2),
        "prediction": prediction,
        "status": (
            "Failure Risk Detected" if prediction == 1 else "Normal"
        ),
    }


# ============================================================
# 5. SINGLE-MACHINE SHAP EXPLANATION
# ============================================================

def explain_machine_prediction(
    machine_type,
    air_temperature,
    process_temperature,
    rotational_speed,
    torque,
    tool_wear,
):
    """
    Explain one prediction with SHAP contributions.

    SHAP values are contributions to the model output (raw margin
    for this tree explainer configuration), not probability changes.
    """
    machine_record = _validate_single_machine(
        machine_type,
        air_temperature,
        process_temperature,
        rotational_speed,
        torque,
        tool_wear,
    )

    machine_data = pd.DataFrame(
        [machine_record],
        columns=FEATURE_COLUMNS,
    )

    transformed_input = preprocessor.transform(machine_data)
    shap_explanation = get_shap_explainer()(transformed_input)

    shap_values = np.asarray(shap_explanation.values)

    # Handle classifiers/explainers that return a class dimension.
    if shap_values.ndim == 3:
        shap_values = shap_values[0, :, 1]
    elif shap_values.ndim == 2:
        shap_values = shap_values[0]
    else:
        raise ValueError(
            f"Unexpected SHAP output shape: {shap_values.shape}"
        )

    feature_names = preprocessor.get_feature_names_out()

    if len(feature_names) != len(shap_values):
        raise ValueError(
            "SHAP values do not match the transformed feature names."
        )

    contributions = []

    for name, value in zip(feature_names, shap_values):
        readable_name = (
            str(name)
            .replace("num__", "")
            .replace("cat__", "")
            .replace("remainder__", "")
        )

        readable_name = {
            "Type_H": "Machine Type H",
            "Type_L": "Machine Type L",
            "Type_M": "Machine Type M",
        }.get(readable_name, readable_name)

        value = float(value)

        contributions.append(
            {
                "feature": readable_name,
                "shap_value": value,
                "influence": (
                    "Increases model output"
                    if value > 0
                    else "Reduces model output"
                    if value < 0
                    else "No contribution"
                ),
            }
        )

    contributions.sort(
        key=lambda item: abs(item["shap_value"]),
        reverse=True,
    )

    return contributions


# ============================================================
# 6. MULTIPLE-MACHINE PREDICTION
# ============================================================

def predict_multiple_machines(machine_data):
    """Validate and predict failure risk for multiple machines."""
    clean_data = _validate_machine_dataframe(machine_data)

    probabilities = final_model.predict_proba(clean_data)[:, 1]
    predictions = (probabilities >= best_threshold).astype(int)

    results = clean_data.copy()
    results["Failure Probability (%)"] = (probabilities * 100).round(2)
    results["Prediction"] = predictions
    results["Machine Status"] = np.where(
        predictions == 1,
        "Failure Risk Detected",
        "Normal",
    )

    return results


# ============================================================
# 7. CSV PREDICTION
# ============================================================

def predict_from_csv(file_path, save_output=True):
    """
    Read a CSV, predict all machine records, and optionally save results.

    The CSV must contain all six FEATURE_COLUMNS. Additional columns
    are preserved in the returned results.
    """
    csv_path = Path(file_path)
    if not csv_path.is_absolute():
        csv_path = BASE_DIR / csv_path

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    machine_data = pd.read_csv(csv_path)

    if machine_data.empty:
        raise ValueError("The CSV file contains no records.")

    clean_features = _validate_machine_dataframe(machine_data)
    prediction_results = predict_multiple_machines(clean_features)

    results = machine_data.copy()
    results["Failure Probability (%)"] = (
        prediction_results["Failure Probability (%)"].to_numpy()
    )
    results["Prediction"] = prediction_results["Prediction"].to_numpy()
    results["Machine Status"] = (
        prediction_results["Machine Status"].to_numpy()
    )

    if save_output:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / "machine_predictions.csv"
        results.to_csv(output_path, index=False)
        print(f"Prediction report saved to: {output_path}")

    print("\nPrediction completed successfully!")
    print(f"Total machines: {len(results)}")
    print(f"Normal machines: {(results['Prediction'] == 0).sum()}")
    print(
        "Machines with failure risk: "
        f"{(results['Prediction'] == 1).sum()}"
    )

    return results


# ============================================================
# 8. MODEL EVALUATION
# ============================================================

def evaluate_csv_predictions(results):
    """
    Evaluate predictions only when the CSV contains ground-truth labels.

    Returns a dictionary so Streamlit can display the metrics directly.
    """
    required_columns = [
        "Machine failure",
        "Prediction",
        "Failure Probability (%)",
    ]
    missing_columns = [
        column for column in required_columns
        if column not in results.columns
    ]

    if missing_columns:
        raise ValueError(
            "Cannot evaluate without these columns: "
            f"{missing_columns}"
        )

    y_true = pd.to_numeric(results["Machine failure"], errors="raise")
    y_pred = pd.to_numeric(results["Prediction"], errors="raise")
    y_probability = (
        pd.to_numeric(
            results["Failure Probability (%)"],
            errors="raise",
        ) / 100
    )

    if not y_true.isin([0, 1]).all():
        raise ValueError("'Machine failure' must contain only 0 or 1.")

    if not y_pred.isin([0, 1]).all():
        raise ValueError("'Prediction' must contain only 0 or 1.")

    if not y_probability.between(0, 1).all():
        raise ValueError("Failure probabilities must be between 0 and 100%.")

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["Normal", "Failure"],
        output_dict=True,
        zero_division=0,
    )
    pr_auc = float(average_precision_score(y_true, y_probability))

    evaluation = {
        "confusion_matrix": matrix,
        "classification_report": report,
        "pr_auc": pr_auc,
        "actual_class_distribution": y_true.value_counts().sort_index(),
    }

    print("\nPREDICTION PERFORMANCE REPORT")
    print("=" * 40)
    print("\nConfusion Matrix:")
    print(matrix)
    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["Normal", "Failure"],
            zero_division=0,
        )
    )
    print(f"PR-AUC: {pr_auc:.4f}")

    return evaluation


# ============================================================
# 9. AUTOMATED FAILURE ANALYSIS
# ============================================================

def generate_failure_analysis(results, top_n=10, save_output=True):
    """
    Summarize predicted risk and rank machines by predicted probability.

    The risk percentage is the percentage flagged by the model; it is
    not the observed failure rate unless ground-truth labels are used.
    """
    required_columns = [
        "Failure Probability (%)",
        "Prediction",
        "Machine Status",
    ]
    missing_columns = [
        column for column in required_columns
        if column not in results.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if results.empty:
        raise ValueError("No machine records available for analysis.")

    if not isinstance(top_n, int) or top_n < 1:
        raise ValueError("top_n must be a positive integer.")

    analysis_data = results.copy()
    analysis_data["Prediction"] = pd.to_numeric(
        analysis_data["Prediction"],
        errors="raise",
    )
    analysis_data["Failure Probability (%)"] = pd.to_numeric(
        analysis_data["Failure Probability (%)"],
        errors="raise",
    )

    if not analysis_data["Prediction"].isin([0, 1]).all():
        raise ValueError("Prediction must contain only 0 or 1.")

    if not analysis_data["Failure Probability (%)"].between(0, 100).all():
        raise ValueError("Failure probabilities must be between 0 and 100.")

    total_machines = len(analysis_data)
    failure_risk_count = int((analysis_data["Prediction"] == 1).sum())
    normal_count = int((analysis_data["Prediction"] == 0).sum())
    risk_percentage = failure_risk_count / total_machines * 100

    summary_df = pd.DataFrame(
        [
            ("Total Machines Analysed", total_machines),
            ("Machines With Failure Risk", failure_risk_count),
            ("Normal Machines", normal_count),
            ("Failure Risk Percentage", round(risk_percentage, 2)),
        ],
        columns=["Metric", "Value"],
    )

    # These are row numbers in the supplied results, not physical asset IDs.
    if "Machine Record" not in analysis_data.columns:
        analysis_data.insert(
            0,
            "Machine Record",
            np.arange(1, total_machines + 1),
        )

    top_risk_machines = (
        analysis_data.loc[analysis_data["Prediction"] == 1]
        .sort_values(
            by="Failure Probability (%)",
            ascending=False,
        )
        .head(top_n)
        .reset_index(drop=True)
    )

    if save_output:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(
            OUTPUT_DIR / "failure_analysis_summary.csv",
            index=False,
        )
        top_risk_machines.to_csv(
            OUTPUT_DIR / "top_risk_machines.csv",
            index=False,
        )

    print("\n========== FAILURE ANALYSIS REPORT ==========")
    print(f"Total Machines Analysed: {total_machines}")
    print(f"Machines With Failure Risk: {failure_risk_count}")
    print(f"Normal Machines: {normal_count}")
    print(f"Predicted Failure Risk: {risk_percentage:.2f}%")
    print(f"\n========== TOP {top_n} HIGH-RISK MACHINES ==========")

    if top_risk_machines.empty:
        print("No machines were flagged for failure risk.")
    else:
        display_columns = [
            "Machine Record",
            "Failure Probability (%)",
            "Machine Status",
        ]
        print(top_risk_machines[display_columns].to_string(index=False))

    if save_output:
        print("\nFailure analysis reports saved in outputs/.")

    return summary_df, top_risk_machines


# ============================================================
# 10. FAILURE PATTERN ANALYSIS
# ============================================================

def generate_failure_pattern_analysis(results, save_output=True):
    """
    Compare input-feature statistics across model-predicted groups
    and summarize predicted risk by machine type.

    These are descriptive associations, not causal explanations.
    """
    required_columns = (
        NUMERIC_COLUMNS
        + ["Type", "Prediction", "Failure Probability (%)"]
    )
    missing_columns = [
        column for column in required_columns
        if column not in results.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if results.empty:
        raise ValueError("No machine records available for analysis.")

    analysis_data = results.copy()

    for column in NUMERIC_COLUMNS + [
        "Prediction",
        "Failure Probability (%)",
    ]:
        analysis_data[column] = pd.to_numeric(
            analysis_data[column],
            errors="raise",
        )

    if not analysis_data["Type"].isin(VALID_MACHINE_TYPES).all():
        raise ValueError("Machine Type must contain only L, M, or H.")

    if not analysis_data["Prediction"].isin([0, 1]).all():
        raise ValueError("Prediction must contain only 0 or 1.")

    if not analysis_data["Failure Probability (%)"].between(0, 100).all():
        raise ValueError("Failure probabilities must be between 0 and 100.")

    analysis_data["Predicted Status"] = analysis_data["Prediction"].map(
        {0: "Normal", 1: "Failure Risk"}
    )

    # 1. Numeric feature statistics by predicted status.
    pattern_summary = (
        analysis_data
        .groupby("Predicted Status")[NUMERIC_COLUMNS]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .round(2)
    )
    pattern_summary.columns = [
        f"{feature}_{statistic}"
        for feature, statistic in pattern_summary.columns
    ]
    pattern_summary = pattern_summary.reset_index()

    # 2. Predicted risk by machine type.
    type_analysis = (
        analysis_data
        .groupby("Type")
        .agg(
            Total_Machines=("Prediction", "count"),
            Failure_Risk_Machines=("Prediction", "sum"),
            Average_Failure_Probability=(
                "Failure Probability (%)",
                "mean",
            ),
        )
        .reset_index()
    )

    type_analysis["Normal_Machines"] = (
        type_analysis["Total_Machines"]
        - type_analysis["Failure_Risk_Machines"]
    )
    type_analysis["Failure_Risk_Rate (%)"] = (
        type_analysis["Failure_Risk_Machines"]
        / type_analysis["Total_Machines"]
        * 100
    )
    type_analysis = type_analysis.round(2)

    # 3. Mean feature values by predicted status.
    feature_comparison = (
        analysis_data
        .groupby("Predicted Status")[NUMERIC_COLUMNS]
        .mean()
        .round(2)
        .reset_index()
    )

    if save_output:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        pattern_summary.to_csv(
            OUTPUT_DIR / "failure_pattern_summary.csv",
            index=False,
        )
        type_analysis.to_csv(
            OUTPUT_DIR / "failure_pattern_by_machine_type.csv",
            index=False,
        )
        feature_comparison.to_csv(
            OUTPUT_DIR / "failure_feature_comparison.csv",
            index=False,
        )

    print("\n========== FAILURE PATTERN ANALYSIS ==========")
    print("\n1. NUMERIC FEATURE STATISTICS")
    print(pattern_summary.to_string(index=False))
    print("\n2. FAILURE RISK BY MACHINE TYPE")
    print(type_analysis.to_string(index=False))
    print("\n3. AVERAGE FEATURE COMPARISON")
    print(feature_comparison.to_string(index=False))

    if save_output:
        print("\nFailure pattern reports saved in outputs/.")

    return pattern_summary, type_analysis, feature_comparison


# ============================================================
# 11. COMMAND-LINE SMOKE TESTS
# ============================================================

def run_smoke_tests():
    """Run the original example tests when this file is executed directly."""
    print("Model loaded successfully!")
    print("Threshold:", best_threshold)
    print("Pipeline steps:", list(final_model.named_steps.keys()))

    # Single-machine prediction.
    test_result = predict_machine(
        machine_type="M",
        air_temperature=298.1,
        process_temperature=308.6,
        rotational_speed=1500,
        torque=40,
        tool_wear=120,
    )

    print("\nMACHINE PREDICTION")
    print("-------------------------")
    print("Failure Probability:", test_result["failure_percentage"], "%")
    print("Prediction:", test_result["prediction"])
    print("Machine Status:", test_result["status"])

    # Single-machine SHAP explanation.
    explanation = explain_machine_prediction(
        machine_type="M",
        air_temperature=298.1,
        process_temperature=308.6,
        rotational_speed=1500,
        torque=40,
        tool_wear=120,
    )

    print("\nMACHINE PREDICTION EXPLANATION")
    print("--------------------------------")
    for item in explanation:
        print(
            item["feature"],
            "| SHAP Value:", round(item["shap_value"], 4),
            "|", item["influence"],
        )

    # Batch prediction.
    sample_machines = pd.DataFrame(
        [
            {
                "Type": "M",
                "Air temperature [K]": 298.1,
                "Process temperature [K]": 308.6,
                "Rotational speed [rpm]": 1500,
                "Torque [Nm]": 40,
                "Tool wear [min]": 120,
            },
            {
                "Type": "L",
                "Air temperature [K]": 300.0,
                "Process temperature [K]": 310.0,
                "Rotational speed [rpm]": 1400,
                "Torque [Nm]": 50,
                "Tool wear [min]": 200,
            },
        ]
    )

    batch_results = predict_multiple_machines(sample_machines)
    print("\nMULTIPLE MACHINE PREDICTIONS")
    print(batch_results.to_string(index=False))

    # CSV workflow and analysis run only if the example CSV exists.
    test_csv = BASE_DIR / "independent_test_data.csv"

    if not test_csv.exists():
        print(
            "\nSkipping CSV workflow: independent_test_data.csv "
            "was not found in the project root."
        )
        return

    csv_results = predict_from_csv(test_csv)
    print("\nFIRST 10 PREDICTION RESULTS")
    print(csv_results.head(10).to_string(index=False))

    if "Machine failure" in csv_results.columns:
        evaluate_csv_predictions(csv_results)
    else:
        print(
            "\nSkipping model evaluation: CSV has no "
            "'Machine failure' ground-truth column."
        )

    generate_failure_analysis(csv_results)
    generate_failure_pattern_analysis(csv_results)


if __name__ == "__main__":
    run_smoke_tests()
