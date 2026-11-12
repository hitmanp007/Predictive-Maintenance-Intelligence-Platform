# Predictive Maintenance Intelligence Platform

An end-to-end machine-learning research prototype for estimating
machine-failure risk from operating parameters. It combines an XGBoost
model, a reusable Python prediction engine, a Streamlit dashboard, batch
CSV analysis, and SHAP-based explainability.

**Status:** Working prototype\
**Dataset:** UCI AI4I 2020 Predictive Maintenance Dataset\
**Model:** XGBoost classifier\
**Interface:** Streamlit\
**Explainability:** SHAP

## Table of contents

-   [Overview](#overview)
-   [Objectives](#objectives)
-   [Features](#features)
-   [Dataset and input features](#dataset-and-input-features)
-   [Data preparation](#data-preparation)
-   [Model development and results](#model-development-and-results)
-   [Explainable AI](#explainable-ai)
-   [Dashboard modules](#dashboard-modules)
-   [Project structure](#project-structure)
-   [Technology stack](#technology-stack)
-   [Installation and execution](#installation-and-execution)
-   [Input CSV format](#input-csv-format)
-   [Generated reports](#generated-reports)
-   [Testing status](#testing-status)
-   [Limitations](#limitations)
-   [Future improvements](#future-improvements)
-   [References](#references)

## Overview

Unexpected equipment failures can interrupt industrial operations and
lead to downtime and maintenance costs. Predictive maintenance uses
operational measurements and analytical models to identify patterns
associated with possible failures.

This project predicts machine-failure probability from six operational
features. Users can analyse one machine or upload a CSV of multiple
records, inspect predicted risk, review failure-pattern summaries, and
generate SHAP explanations.

The application is an educational and portfolio research prototype. It
is not a certified industrial monitoring or safety system.

## Objectives

-   Develop an end-to-end machine-failure classification workflow.
-   Compare multiple classification algorithms.
-   Evaluate performance on an imbalanced target.
-   Select a classification threshold using validation data.
-   Support single-machine and batch predictions.
-   Explain model outputs using SHAP.
-   Present results in an interactive dashboard.
-   Generate downloadable analysis reports.

## Features

-   **Single prediction:** Enter machine parameters and receive a
    predicted failure probability and class.
-   **Batch prediction:** Upload CSV records, predict all machines,
    filter results, and download a report.
-   **Failure analysis:** Summarise model-flagged records and rank
    records by predicted probability.
-   **Failure patterns:** Compare predictions by machine type and
    inspect operating-feature summaries.
-   **Explainable AI:** View per-record SHAP feature contributions in a
    table and chart.
-   **Model evaluation:** When actual labels are supplied, calculate
    classification metrics and display the confusion matrix.

## Dataset and input features

The project uses the [UCI AI4I 2020 Predictive Maintenance
Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset).

-   10,000 records and 14 original columns.
-   Target: `Machine failure` (0 = no failure, 1 = failure).
-   Synthetic predictive-maintenance dataset.

The model uses these six input features:

  -----------------------------------------------------------------------
  Feature                             Description
  ----------------------------------- -----------------------------------
  `Type`                              Product quality type: L, M, or H

  `Air temperature [K]`               Air temperature in kelvin

  `Process temperature [K]`           Process temperature in kelvin

  `Rotational speed [rpm]`            Rotational speed in revolutions per
                                      minute

  `Torque [Nm]`                       Torque in newton-metres

  `Tool wear [min]`                   Tool wear in minutes
  -----------------------------------------------------------------------

### Target distribution

  Class              Records      Share
  ------------- ------------ ----------
  Normal (0)           9,661     96.61%
  Failure (1)            339      3.39%
  **Total**       **10,000**   **100%**

The target is highly imbalanced, so accuracy is considered alongside
failure-class precision, recall, F1-score, PR-AUC, and the confusion
matrix.

## Data preparation

-   `UDI` and `Product ID` are excluded because they are identifiers,
    not operational measurements.
-   `TWF`, `HDF`, `PWF`, `OSF`, and `RNF` are excluded from model inputs
    to avoid leakage from failure-mode indicators.
-   Numeric features are transformed using `StandardScaler`.
-   `Type` is one-hot encoded using
    `OneHotEncoder(handle_unknown="ignore")`.
-   A scikit-learn `ColumnTransformer` and classifier are persisted
    together as a pipeline, so inference uses the training
    transformations.

## Model development and results

Four algorithms were evaluated:

1.  Logistic Regression
2.  Decision Tree
3.  Random Forest
4.  XGBoost

The final model is an XGBoost classifier inside the saved preprocessing
pipeline.

### Cross-validation results

  Metric          Mean   Standard deviation
  ----------- -------- --------------------
  PR-AUC        0.7389               0.0434
  ROC-AUC       0.9691               0.0155
  Precision     0.5168               0.0292
  Recall        0.7693               0.0510
  F1-score      0.6172               0.0259

### Decision threshold

A decision threshold was selected using validation data rather than
assuming 0.50.

-   Selected threshold: `0.7593754` (approximately).
-   Saved in `models/xgboost_threshold.pkl`.
-   The trained pipeline is saved in
    `models/xgboost_predictive_maintenance.pkl`.

The threshold converts predicted probabilities into class labels.
Changing it can change the false-positive and false-negative trade-off.

### Final holdout evaluation

The final evaluation used an untouched 20% holdout split containing
2,000 records.

  Metric                Result
  ------------------- --------
  Accuracy               97.8%
  Failure precision       0.72
  Failure recall          0.75
  Failure F1-score        0.73
  PR-AUC                0.7991
  ROC-AUC               0.9716

**Confusion matrix** (rows = actual class; columns = predicted class):

                     Predicted normal   Predicted failure
  ---------------- ------------------ -------------------
  Actual normal                 1,912                  20
  Actual failure                   17                  51

-   True negatives: 1,912
-   False positives: 20
-   False negatives: 17
-   True positives: 51

The model identified 51 of 68 failure records in this holdout set
(failure recall 0.75). There were 20 false-positive predictions among
1,932 normal records.

**Evaluation clarification:** `independent_test_data.csv` contains this
same 2,000-row holdout split. It is not a second independently collected
dataset. Do not describe it as an additional independent validation.

## Explainable AI

The project uses [SHAP](https://shap.readthedocs.io/) with the XGBoost
classifier to inspect feature contributions for individual predictions.

### Recorded global SHAP importance

  Feature                 Recorded importance
  --------------------- ---------------------
  Torque                                 1.99
  Tool wear                              1.69
  Rotational speed                       1.24
  Air temperature                        1.12
  Process temperature                    0.42
  Type M                                 0.08
  Type L                                 0.06
  Type H                                 0.05

These are the recorded global importance values from the project's SHAP
analysis. They are not physical risk scores and do not establish
causation.

For individual explanations: - Positive SHAP values increase the model's
raw output relative to its baseline. - Negative SHAP values reduce the
model's raw output. - The values are on the model's raw margin/log-odds
scale, not percentage-point changes in failure probability.

SHAP explains model behaviour; it does not prove why a physical machine
fails.

## Dashboard modules

  -----------------------------------------------------------------------
  Module                              Function
  ----------------------------------- -----------------------------------
  Overview                            Batch statistics, predicted-risk
                                      share, highest-risk records

  Single Machine Prediction           Individual input form and
                                      prediction

  Batch Prediction                    CSV upload, predictions, filtering,
                                      downloads

  Failure Analysis                    Summary and highest-risk records

  Failure Patterns                    Comparison by machine type and
                                      operating features

  Explainable AI                      Individual SHAP contributions and
                                      chart

  Model Performance                   Metrics and confusion matrix when
                                      ground-truth labels exist
  -----------------------------------------------------------------------

The latest batch is retained in Streamlit session state for use by the
other dashboard pages.

## System workflow

``` text
Machine operating data
        |
        v
Input validation
        |
        v
Saved preprocessing pipeline
(scaling + one-hot encoding)
        |
        v
     XGBoost
        |
        v
Failure probability
        |
        v
Saved decision threshold
        |
   +----+----+
   |         |
Normal    Predicted risk
   |         |
   +----+----+
        |
        v
 Streamlit dashboard
        |
 +------+------+------+
 |             |      |
Batch       Analysis  SHAP
reports     summaries explanations
```

## Project structure

``` text
Predictive_Maintenance/
│
├── app.py
├── prediction_engine.py
├── explainability.py
├── independent_test_data.csv
├── requirements.txt
│
├── models/
│   ├── xgboost_predictive_maintenance.pkl
│   └── xgboost_threshold.pkl
│
└── outputs/
    ├── machine_predictions.csv
    ├── failure_analysis_summary.csv
    ├── top_risk_machines.csv
    ├── failure_pattern_summary.csv
    ├── failure_pattern_by_machine_type.csv
    ├── failure_feature_comparison.csv
    └── high_risk_machine_shap_explanations.csv
```

Output files are generated when the relevant analysis functions are run;
they may not exist in a fresh clone.

### Main files

-   `app.py`: Streamlit dashboard.
-   `prediction_engine.py`: model loading, input validation,
    single/batch inference, evaluation, and failure analysis.
-   `explainability.py`: SHAP explanation workflow and explanation
    report.
-   `models/`: saved pipeline and decision threshold.
-   `outputs/`: generated prediction and analysis files.
-   `independent_test_data.csv`: existing 2,000-row holdout data.

## Technology stack

  Area                Tools
  ------------------- --------------------------------------------------
  Language            Python
  Data handling       Pandas, NumPy
  Machine learning    scikit-learn, XGBoost
  Preprocessing       StandardScaler, OneHotEncoder, ColumnTransformer
  Model persistence   Joblib
  Explainability      SHAP
  Dashboard           Streamlit
  Data exchange       CSV
  Development         Jupyter Notebook, VS Code

## Installation and execution

### 1. Clone the repository

``` bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Predictive_Maintenance
```

Replace the placeholder with the actual repository URL.

### 2. Create a virtual environment

Windows PowerShell:

``` powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

If `requirements.txt` is available:

``` bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Otherwise, install the main packages:

``` bash
python -m pip install streamlit pandas numpy scikit-learn xgboost shap joblib
```

For reproducible deployment, create and test a pinned dependency file
matching the environment used to train the saved model.

### 4. Check model artifacts

Ensure these files exist:

``` text
models/xgboost_predictive_maintenance.pkl
models/xgboost_threshold.pkl
```

### 5. Run the dashboard

``` bash
streamlit run app.py
```

To run the prediction engine's command-line smoke tests:

``` bash
python prediction_engine.py
```

## Input CSV format

Batch prediction requires these six columns:

``` csv
Type,Air temperature [K],Process temperature [K],Rotational speed [rpm],Torque [Nm],Tool wear [min]
M,298.1,308.6,1500,40,120
L,299.0,309.0,1400,35,100
H,297.5,308.0,1600,32,80
```

These rows are illustrative examples, not real factory measurements.

`Type` must be L, M, or H. Numeric values must be finite and satisfy the
validation rules in `prediction_engine.py`.

For evaluation, the CSV must also include the ground-truth
`Machine failure` column. Without actual labels, predictions can be
generated but classification performance cannot be calculated.

## Generated reports

  -------------------------------------------------------------------------------
  File                                        Purpose
  ------------------------------------------- -----------------------------------
  `machine_predictions.csv`                   Input records plus predicted
                                              probability, class, and status

  `failure_analysis_summary.csv`              Summary of analysed and
                                              model-flagged records

  `top_risk_machines.csv`                     Records ranked by predicted failure
                                              probability

  `failure_pattern_summary.csv`               Feature statistics grouped by
                                              predicted status

  `failure_pattern_by_machine_type.csv`       Prediction summary by machine type

  `failure_feature_comparison.csv`            Feature comparison across predicted
                                              groups

  `high_risk_machine_shap_explanations.csv`   SHAP contributions for selected
                                              records
  -------------------------------------------------------------------------------

A generated `Machine Record` value is a row reference in the uploaded
file unless a genuine equipment identifier is provided. It is not
automatically a physical machine ID.

## Testing status

The primary workflows have been manually run:

-   Prediction engine execution.
-   Single-machine prediction.
-   Batch CSV prediction.
-   SHAP explanation generation.
-   CSV report generation and download.
-   Streamlit dashboard navigation and use.
-   Baseline holdout confusion-matrix verification.

The recorded baseline confusion matrix was reproduced:

``` text
[[1912, 20],
 [  17, 51]]
```

Further input-validation and automated pytest coverage are planned.
These manual checks do not constitute a complete automated software test
suite.

## Limitations

1.  **Synthetic dataset:** AI4I 2020 is synthetic. Results on this
    dataset do not establish equivalent performance on real industrial
    equipment.
2.  **No live sensor integration:** The current app processes entered
    values and uploaded CSV files; it does not connect to live
    machinery.
3.  **Not safety-certified:** Predictions must not be the sole basis for
    maintenance, shutdown, or safety-critical decisions.
4.  **Risk is a prediction:** A model-flagged record is not proof that a
    machine has failed or will fail.
5.  **SHAP is not causal:** Feature contributions explain model output,
    not physical failure causes.
6.  **No second independent test set:** The supplied holdout CSV is the
    same split used for the reported results.
7.  **Real-world validation is required:** Deployment would require
    representative plant data, external validation, monitoring, and
    review by qualified domain experts.

## Future improvements

-   Complete unit and integration tests using pytest.
-   Strengthen validation for missing values, malformed CSVs, and
    out-of-range measurements.
-   Add interactive monitoring visualisations.
-   Introduce real machine identifiers and machine-level history.
-   Add time-series ingestion and trend analysis.
-   Integrate industrial IoT or real sensor sources.
-   Add model versioning, prediction logging, and drift monitoring.
-   Evaluate on independently collected industrial data.
-   Deploy with reproducible dependencies.
-   Add authentication and role-based access if multi-user operation is
    needed.

## References

1.  [UCI Machine Learning Repository --- AI4I 2020 Predictive
    Maintenance
    Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)
2.  [XGBoost documentation](https://xgboost.readthedocs.io/)
3.  [scikit-learn documentation](https://scikit-learn.org/stable/)
4.  [SHAP documentation](https://shap.readthedocs.io/)
5.  [Streamlit documentation](https://docs.streamlit.io/)
6.  [Python documentation](https://docs.python.org/3/)

------------------------------------------------------------------------

**Author:** Developed as a machine-learning and predictive analytics
portfolio project.

*This README reflects the current prototype and recorded results. Update
the repository URL, dependency versions, deployment details, and test
status as the project evolves.*
