
import os
import joblib
import pandas as pd
import numpy as np
import shap

# -----------------------------------------
# LOAD TRAINED MODEL
# -----------------------------------------

MODEL_PATH = "models/xgboost_predictive_maintenance.pkl"

final_model = joblib.load(MODEL_PATH)

# Extract preprocessing and XGBoost classifier
preprocessor = final_model.named_steps["preprocessing"]
xgb_classifier = final_model.named_steps["classifier"]

# Initialise SHAP explainer
explainer = shap.TreeExplainer(xgb_classifier)

print("SHAP Explainability Module Loaded Successfully!")

# high risk explaination module



def explain_high_risk_machines(top_risk_machines, top_n=10):
    """
    Generate SHAP explanations for the highest-risk machines.

    Positive SHAP values increase the model output towards
    failure. Negative values decrease it.
    """

    if top_risk_machines.empty:
        print("No high-risk machines available for explanation.")
        return pd.DataFrame()

    required_features = [
        "Type",
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]"
    ]

    missing_columns = [
        col for col in required_features
        if col not in top_risk_machines.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required features: {missing_columns}"
        )

    machines_to_explain = top_risk_machines.head(top_n)

    explanations = []

    print("\n========== SHAP HIGH-RISK MACHINE EXPLANATIONS ==========")

    for _, machine in machines_to_explain.iterrows():

        # Prepare one machine's input
        machine_data = pd.DataFrame(
            [machine[required_features].to_dict()]
        )

        # Apply the same preprocessing used during training
        transformed_data = preprocessor.transform(machine_data)

        # Generate SHAP values
        shap_values = explainer.shap_values(transformed_data)

        shap_values = np.asarray(shap_values)

        # Handle possible SHAP output dimensions
        if shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]

        machine_shap_values = shap_values[0]

        # Get transformed feature names
        feature_names = preprocessor.get_feature_names_out()

        # Rank features by absolute SHAP contribution
        ranked_indices = np.argsort(
            np.abs(machine_shap_values)
        )[::-1]

        print(
            f"\nMachine Record: {machine.get('Machine Record', 'Unknown')}"
        )

        print(
            f"Failure Probability: "
            f"{machine['Failure Probability (%)']:.2f}%"
        )

        machine_contributions = []

        for index in ranked_indices:

            feature_name = feature_names[index]
            contribution = float(machine_shap_values[index])

            if contribution > 0:
                impact = "Increases failure prediction"
            elif contribution < 0:
                impact = "Reduces failure prediction"
            else:
                impact = "No contribution"

            print(
                f"{feature_name} | "
                f"SHAP: {contribution:.4f} | {impact}"
            )

            machine_contributions.append({
                "Machine Record": machine.get(
                    "Machine Record", "Unknown"
                ),
                "Failure Probability (%)": machine[
                    "Failure Probability (%)"
                ],
                "Feature": feature_name,
                "SHAP Value": contribution,
                "Impact": impact
            })

        explanations.extend(machine_contributions)

    # Convert explanations to DataFrame
    explanation_df = pd.DataFrame(explanations)

    # Save explanations
    os.makedirs("outputs", exist_ok=True)

    explanation_df.to_csv(
        "outputs/high_risk_machine_shap_explanations.csv",
        index=False
    )

    print("\nSHAP explanations saved successfully!")
    print(
        "Output: outputs/high_risk_machine_shap_explanations.csv"
    )

    return explanation_df



if __name__ == "__main__":

    results = pd.read_csv(
        "outputs/machine_predictions.csv"
    )

    # Assign a record number to each original CSV row
    results.insert(
        0,
        "Machine Record",
        range(1, len(results) + 1)
    )

    # Select the highest-risk machines
    top_risk_machines = (
        results[results["Prediction"] == 1]
        .sort_values(
            by="Failure Probability (%)",
            ascending=False
        )
        .head(10)
        .copy()
    )

    # Generate SHAP explanations
    explanation_df = explain_high_risk_machines(
        top_risk_machines,
        top_n=10
    )
    