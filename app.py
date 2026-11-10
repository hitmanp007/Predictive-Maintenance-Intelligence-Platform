"""
Predictive Maintenance Intelligence Platform
Streamlit dashboard

Run from the project root:
    streamlit run app.py

Required project files:
    prediction_engine.py
    models/xgboost_predictive_maintenance.pkl
    models/xgboost_threshold.pkl
"""

import io
import sys
import traceback
from pathlib import Path

import pandas as pd
import streamlit as st


# ------------------------------------------------------------
# 1. PAGE CONFIGURATION
# ------------------------------------------------------------

st.set_page_config(
    page_title="Predictive Maintenance Intelligence",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


# ------------------------------------------------------------
# 2. IMPORT PREDICTION ENGINE
# ------------------------------------------------------------

try:
    from prediction_engine import (
        FEATURE_COLUMNS,
        NUMERIC_COLUMNS,
        best_threshold,
        predict_machine,
        predict_multiple_machines,
        explain_machine_prediction,
        evaluate_csv_predictions,
        generate_failure_analysis,
        generate_failure_pattern_analysis,
    )

    ENGINE_ERROR = None

except Exception as exc:
    ENGINE_ERROR = exc


# ------------------------------------------------------------
# 3. STYLING
# ------------------------------------------------------------

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.7rem;
            padding-bottom: 2.5rem;
        }
        div[data-testid="stMetric"] {
            background: rgba(128, 128, 128, 0.08);
            border: 1px solid rgba(128, 128, 128, 0.18);
            padding: 14px 16px;
            border-radius: 12px;
        }
        .section-note {
            color: #8b929d;
            font-size: 0.9rem;
        }
        div.stButton > button {
            border-radius: 8px;
            font-weight: 600;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# 4. HELPER FUNCTIONS
# ------------------------------------------------------------

def show_engine_error():
    """Show a useful message if the prediction engine cannot load."""
    st.error("The prediction engine could not be loaded.")
    st.code(str(ENGINE_ERROR))
    st.info(
        "Check that prediction_engine.py and the models/ folder are "
        "in the same project directory as app.py."
    )


def validate_uploaded_csv(uploaded_file):
    """Read an uploaded CSV and check that model features are present."""
    data = pd.read_csv(uploaded_file)

    if data.empty:
        raise ValueError("The uploaded CSV contains no records.")

    missing = [column for column in FEATURE_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required model columns: {missing}")

    return data


def run_batch_prediction(data):
    """Run predictions and retain original uploaded columns."""
    predictions = predict_multiple_machines(data[FEATURE_COLUMNS].copy())

    results = data.copy()
    results["Failure Probability (%)"] = (
        predictions["Failure Probability (%)"].to_numpy()
    )
    results["Prediction"] = predictions["Prediction"].to_numpy()
    results["Machine Status"] = predictions["Machine Status"].to_numpy()

    if "Machine Record" not in results.columns:
        results.insert(0, "Machine Record", range(1, len(results) + 1))

    return results


def store_results(results, source_name):
    """Save the latest batch result in Streamlit session state."""
    st.session_state["batch_results"] = results
    st.session_state["batch_source"] = source_name


def get_results():
    return st.session_state.get("batch_results")


def metric_card(label, value, help_text=None):
    st.metric(label=label, value=value, help=help_text)


def show_prediction_status(prediction, probability):
    if prediction["prediction"] == 1:
        st.error(
            f"Failure risk detected — model probability: "
            f"{probability:.2f}%"
        )
    else:
        st.success(
            f"Normal prediction — model probability: "
            f"{probability:.2f}%"
        )

    st.progress(min(max(float(probability) / 100, 0.0), 1.0))


def display_download_button(dataframe, filename, label="Download CSV report"):
    csv_bytes = dataframe.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=label,
        data=csv_bytes,
        file_name=filename,
        mime="text/csv",
        use_container_width=False,
    )


# ------------------------------------------------------------
# 5. HEADER AND SIDEBAR NAVIGATION
# ------------------------------------------------------------

st.title("⚙️ Predictive Maintenance Intelligence")
st.caption(
    "Machine failure risk prediction • Batch monitoring • "
    "Failure pattern analysis • Explainable AI"
)

with st.sidebar:
    st.title("⚙️ PM Intelligence")
    st.caption("Predictive Maintenance Platform")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "Single Machine Prediction",
            "Batch Prediction",
            "Failure Analysis",
            "Failure Patterns",
            "Explainable AI",
            "Model Performance",
        ],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("**Model configuration**")
    st.write(f"Decision threshold: `{best_threshold:.4f}`" if ENGINE_ERROR is None else "Model unavailable")
    st.caption("Predictions use the saved XGBoost pipeline.")

    if st.button("Clear current batch results", use_container_width=True):
        st.session_state.pop("batch_results", None)
        st.session_state.pop("batch_source", None)
        st.rerun()


if ENGINE_ERROR is not None:
    show_engine_error()
    st.stop()


# ------------------------------------------------------------
# 6. OVERVIEW
# ------------------------------------------------------------

if page == "Overview":
    st.subheader("Platform overview")
    st.write(
        "Use this dashboard to estimate machine failure risk from "
        "operating parameters and inspect model-generated patterns."
    )

    results = get_results()

    if results is None:
        st.info(
            "No batch has been analysed in this session yet. "
            "Open **Batch Prediction** and upload a CSV to populate the dashboard."
        )

        col1, col2, col3 = st.columns(3)
        col1.metric("Model", "XGBoost")
        col2.metric("Input features", len(FEATURE_COLUMNS))
        col3.metric("Decision threshold", f"{best_threshold:.3f}")

        st.markdown("#### Required input columns")
        st.dataframe(
            pd.DataFrame({"Required model feature": FEATURE_COLUMNS}),
            hide_index=True,
            use_container_width=True,
        )

    else:
        total = len(results)
        risk_count = int((results["Prediction"] == 1).sum())
        normal_count = total - risk_count
        risk_share = risk_count / total * 100 if total else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Machines analysed", f"{total:,}")
        c2.metric("Model-flagged risk", f"{risk_count:,}")
        c3.metric("Predicted normal", f"{normal_count:,}")
        c4.metric("Flagged share", f"{risk_share:.2f}%")

        st.caption(
            "Flagged share is the percentage classified as high risk by the model. "
            "It is not the observed machine failure rate."
        )

        left, right = st.columns([1, 1])
        with left:
            st.markdown("#### Prediction distribution")
            distribution = (
                results["Machine Status"]
                .value_counts()
                .rename_axis("Status")
                .to_frame("Machines")
            )
            st.bar_chart(distribution)

        with right:
            st.markdown("#### Highest predicted risk")
            top = results.sort_values(
                "Failure Probability (%)", ascending=False
            ).head(8)
            display_cols = [
                col for col in [
                    "Machine Record",
                    "Type",
                    "Failure Probability (%)",
                    "Machine Status",
                ]
                if col in top.columns
            ]
            st.dataframe(
                top[display_cols],
                hide_index=True,
                use_container_width=True,
            )

        st.markdown("#### Latest batch")
        st.caption(f"Source: {st.session_state.get('batch_source', 'Uploaded data')}")
        st.dataframe(results.head(20), hide_index=True, use_container_width=True)


# ------------------------------------------------------------
# 7. SINGLE MACHINE PREDICTION
# ------------------------------------------------------------

elif page == "Single Machine Prediction":
    st.subheader("Single machine prediction")
    st.write("Enter the operating parameters of one machine.")

    with st.form("single_machine_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            machine_type = st.selectbox("Machine type", ["L", "M", "H"], index=1)
            air_temperature = st.number_input(
                "Air temperature (K)", min_value=0.1, value=298.1, step=0.1
            )

        with col2:
            process_temperature = st.number_input(
                "Process temperature (K)", min_value=0.1, value=308.6, step=0.1
            )
            rotational_speed = st.number_input(
                "Rotational speed (rpm)", min_value=0.1, value=1500.0, step=10.0
            )

        with col3:
            torque = st.number_input(
                "Torque (Nm)", min_value=0.0, value=40.0, step=1.0
            )
            tool_wear = st.number_input(
                "Tool wear (min)", min_value=0.0, value=120.0, step=1.0
            )

        submitted = st.form_submit_button(
            "Predict machine risk",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        try:
            result = predict_machine(
                machine_type=machine_type,
                air_temperature=air_temperature,
                process_temperature=process_temperature,
                rotational_speed=rotational_speed,
                torque=torque,
                tool_wear=tool_wear,
            )

            st.divider()
            st.markdown("### Prediction result")

            result_col1, result_col2 = st.columns(2)
            result_col1.metric(
                "Failure probability",
                f"{result['failure_percentage']:.2f}%",
            )
            result_col2.metric(
                "Model decision",
                result["status"],
            )

            show_prediction_status(
                result,
                result["failure_percentage"],
            )

            st.caption(
                f"Classified using the saved decision threshold "
                f"of {best_threshold:.4f}."
            )

            st.session_state["last_single_input"] = {
                "machine_type": machine_type,
                "air_temperature": air_temperature,
                "process_temperature": process_temperature,
                "rotational_speed": rotational_speed,
                "torque": torque,
                "tool_wear": tool_wear,
            }
            st.session_state["last_single_result"] = result

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

    if "last_single_result" in st.session_state:
        st.divider()
        st.markdown("### Explain this prediction")
        if st.button("Generate SHAP explanation", key="single_shap"):
            try:
                values = st.session_state["last_single_input"]
                explanation = explain_machine_prediction(**values)
                explanation_df = pd.DataFrame(explanation)

                st.dataframe(
                    explanation_df,
                    hide_index=True,
                    use_container_width=True,
                )

                chart_data = explanation_df.set_index("feature")[["shap_value"]]
                st.markdown(
                    "Positive SHAP values increase the model output; "
                    "negative values reduce it. These are model-output "
                    "contributions, not probability changes or causal effects."
                )
                st.bar_chart(chart_data)

            except Exception as exc:
                st.error(f"Could not generate SHAP explanation: {exc}")


# ------------------------------------------------------------
# 8. BATCH PREDICTION
# ------------------------------------------------------------

elif page == "Batch Prediction":
    st.subheader("Batch machine prediction")
    st.write(
        "Upload a CSV containing the six model input features. "
        "Additional columns are preserved in the output."
    )

    st.markdown("#### Required CSV columns")
    st.code(", ".join(FEATURE_COLUMNS), language="text")

    uploaded_file = st.file_uploader(
        "Choose machine data CSV",
        type=["csv"],
        key="batch_csv_uploader",
    )

    if uploaded_file is not None:
        try:
            raw_data = validate_uploaded_csv(uploaded_file)

            st.markdown("#### Uploaded data preview")
            st.caption(f"{len(raw_data):,} machine records")
            st.dataframe(raw_data.head(10), hide_index=True, use_container_width=True)

            if st.button(
                "Run batch predictions",
                type="primary",
                use_container_width=True,
            ):
                with st.spinner("Analysing machine records..."):
                    results = run_batch_prediction(raw_data)
                    store_results(results, uploaded_file.name)

                st.success(f"Predictions completed for {len(results):,} records.")

        except Exception as exc:
            st.error(f"Could not process this CSV: {exc}")

    results = get_results()
    if results is not None:
        st.divider()
        st.markdown("#### Batch prediction results")

        risk_count = int((results["Prediction"] == 1).sum())
        normal_count = len(results) - risk_count
        a, b, c = st.columns(3)
        a.metric("Total records", f"{len(results):,}")
        b.metric("Flagged as risk", f"{risk_count:,}")
        c.metric("Predicted normal", f"{normal_count:,}")

        status_filter = st.selectbox(
            "Filter results",
            ["All machines", "Failure Risk Detected", "Normal"],
        )

        filtered = results
        if status_filter != "All machines":
            filtered = results[results["Machine Status"] == status_filter]

        st.dataframe(
            filtered,
            hide_index=True,
            use_container_width=True,
        )
        display_download_button(
            results,
            "machine_predictions.csv",
            "Download complete prediction report",
        )


# ------------------------------------------------------------
# 9. FAILURE ANALYSIS
# ------------------------------------------------------------

elif page == "Failure Analysis":
    st.subheader("Failure risk analysis")
    results = get_results()

    if results is None:
        st.info("Run a batch prediction first to view failure analysis.")
    else:
        try:
            summary_df, top_risk = generate_failure_analysis(
                results,
                top_n=10,
                save_output=False,
            )

            total = len(results)
            risk_count = int((results["Prediction"] == 1).sum())
            risk_percentage = risk_count / total * 100 if total else 0

            c1, c2, c3 = st.columns(3)
            c1.metric("Machines analysed", f"{total:,}")
            c2.metric("High-risk predictions", f"{risk_count:,}")
            c3.metric("Flagged share", f"{risk_percentage:.2f}%")

            st.caption(
                "These are model-flagged predictions, not confirmed failures."
            )

            st.markdown("#### Summary")
            st.dataframe(summary_df, hide_index=True, use_container_width=True)

            st.markdown("#### Top 10 highest-risk machines")
            if top_risk.empty:
                st.success("No machines were flagged at the current threshold.")
            else:
                st.dataframe(top_risk, hide_index=True, use_container_width=True)
                display_download_button(
                    top_risk,
                    "top_risk_machines.csv",
                    "Download high-risk machine report",
                )

            st.markdown("#### Failure probability distribution")
            st.bar_chart(
                results["Failure Probability (%)"]
                .round(0)
                .value_counts()
                .sort_index()
                .rename("Machine count")
            )

        except Exception as exc:
            st.error(f"Failure analysis failed: {exc}")


# ------------------------------------------------------------
# 10. FAILURE PATTERNS
# ------------------------------------------------------------

elif page == "Failure Patterns":
    st.subheader("Failure pattern analysis")
    results = get_results()

    if results is None:
        st.info("Run a batch prediction first to view failure patterns.")
    else:
        try:
            pattern_summary, type_analysis, feature_comparison = (
                generate_failure_pattern_analysis(
                    results,
                    save_output=False,
                )
            )

            st.markdown("#### Predicted risk by machine type")
            st.dataframe(
                type_analysis,
                hide_index=True,
                use_container_width=True,
            )

            chart_data = type_analysis.set_index("Type")[
                ["Failure_Risk_Rate (%)"]
            ]
            st.bar_chart(chart_data)

            st.markdown("#### Average operating parameters by predicted status")
            st.dataframe(
                feature_comparison,
                hide_index=True,
                use_container_width=True,
            )

            st.markdown("#### Detailed feature statistics")
            st.dataframe(
                pattern_summary,
                hide_index=True,
                use_container_width=True,
            )

            st.caption(
                "These summaries describe associations in the model predictions. "
                "They do not establish that a parameter causes machine failure."
            )

        except Exception as exc:
            st.error(f"Failure pattern analysis failed: {exc}")


# ------------------------------------------------------------
# 11. EXPLAINABLE AI
# ------------------------------------------------------------

elif page == "Explainable AI":
    st.subheader("Explainable AI — SHAP")
    st.write(
        "Inspect how input features contribute to the model output "
        "for an individual machine."
    )

    results = get_results()

    if results is None:
        st.info(
            "Run a batch prediction first, or use the Single Machine Prediction "
            "page to generate an explanation for one machine."
        )
    else:
        if "Machine Record" not in results.columns:
            results = results.copy()
            results.insert(0, "Machine Record", range(1, len(results) + 1))

        candidates = results.sort_values(
            "Failure Probability (%)",
            ascending=False,
        )

        selected_record = st.selectbox(
            "Select machine record",
            candidates["Machine Record"].tolist(),
        )

        selected = candidates[
            candidates["Machine Record"] == selected_record
        ].iloc[0]

        left, right = st.columns(2)
        left.metric(
            "Failure probability",
            f"{selected['Failure Probability (%)']:.2f}%",
        )
        right.metric("Model status", selected["Machine Status"])

        if st.button(
            "Explain selected machine",
            type="primary",
            use_container_width=True,
        ):
            try:
                explanation = explain_machine_prediction(
                    machine_type=selected["Type"],
                    air_temperature=selected["Air temperature [K]"],
                    process_temperature=selected["Process temperature [K]"],
                    rotational_speed=selected["Rotational speed [rpm]"],
                    torque=selected["Torque [Nm]"],
                    tool_wear=selected["Tool wear [min]"],
                )

                explanation_df = pd.DataFrame(explanation)
                st.markdown("#### Feature contributions")
                st.dataframe(
                    explanation_df,
                    hide_index=True,
                    use_container_width=True,
                )

                st.markdown("#### SHAP contribution chart")
                st.bar_chart(
                    explanation_df.set_index("feature")[["shap_value"]]
                )

                st.caption(
                    "Positive SHAP values increase the model's raw output; "
                    "negative values reduce it. SHAP contributions are not "
                    "probability changes and do not prove causation."
                )

            except Exception as exc:
                st.error(f"SHAP explanation failed: {exc}")


# ------------------------------------------------------------
# 12. MODEL PERFORMANCE
# ------------------------------------------------------------

elif page == "Model Performance":
    st.subheader("Model performance evaluation")
    results = get_results()

    if results is None:
        st.info("Upload a labelled CSV in Batch Prediction first.")
    elif "Machine failure" not in results.columns:
        st.warning(
            "The current batch does not contain the 'Machine failure' "
            "ground-truth column. Performance metrics cannot be calculated "
            "without actual labels."
        )
    else:
        st.warning(
            "Metrics below evaluate this uploaded labelled dataset. "
            "Do not interpret them as independent test performance unless "
            "the file is a genuinely untouched test set."
        )

        try:
            evaluation = evaluate_csv_predictions(results)

            report = evaluation["classification_report"]
            normal, failure = st.columns(2)

            normal.metric(
                "Normal precision",
                f"{report['Normal']['precision']:.3f}",
            )
            failure.metric(
                "Failure precision",
                f"{report['Failure']['precision']:.3f}",
            )

            c1, c2, c3 = st.columns(3)
            c1.metric("Failure recall", f"{report['Failure']['recall']:.3f}")
            c2.metric("Failure F1-score", f"{report['Failure']['f1-score']:.3f}")
            c3.metric("PR-AUC", f"{evaluation['pr_auc']:.4f}")

            st.markdown("#### Confusion matrix")
            matrix = pd.DataFrame(
                evaluation["confusion_matrix"],
                index=["Actual Normal", "Actual Failure"],
                columns=["Predicted Normal", "Predicted Failure"],
            )
            st.dataframe(matrix, use_container_width=True)

            st.markdown("#### Classification report")
            report_df = pd.DataFrame(report).T
            st.dataframe(report_df.round(4), use_container_width=True)

        except Exception as exc:
            st.error(f"Model evaluation failed: {exc}")


# ------------------------------------------------------------
# 13. FOOTER
# ------------------------------------------------------------

st.divider()
st.caption(
    "Predictive Maintenance Intelligence Platform | "
    "Research prototype using the UCI AI4I 2020 synthetic dataset. "
    "Model outputs are risk estimates, not confirmed equipment diagnoses."
)
