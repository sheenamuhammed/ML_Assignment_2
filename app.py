import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt


# -------------------------------------------------
# Page Configuration
# -------------------------------------------------

st.set_page_config(
    page_title="Student Grade Classification",
    layout="wide"
)

st.title("Student Grade Classification - ML Model Comparison")

st.write(
    "Upload the test dataset and select a machine learning model "
    "to evaluate its performance."
)


# -------------------------------------------------
# Load Models
# -------------------------------------------------

logistic_model = joblib.load(
    "model/logistic_regression_model.pkl"
)

tree_model = joblib.load(
    "model/decision_tree_model.pkl"
)

knn_model = joblib.load(
    "model/knn_model.pkl"
)

nb_model = joblib.load(
    "model/gaussian_naive_bayes_model.pkl"
)

rf_model = joblib.load(
    "model/random_forest_model.pkl"
)

scaler = joblib.load(
    "model/scaler.pkl"
)


# -------------------------------------------------
# Model Selection
# -------------------------------------------------

model_choice = st.selectbox(
    "Select a Machine Learning Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Gaussian Naive Bayes",
        "Random Forest"
    ]
)


# -------------------------------------------------
# Upload Test Dataset
# -------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload Test Data CSV File",
    type=["csv"]
)


if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Test Data")
    st.dataframe(data.head())

    # Check whether target column exists
    if "GradeClass" not in data.columns:
        st.error(
            "The uploaded dataset must contain the GradeClass column."
        )

    else:

        # Separate features and target
        X_test_app = data.drop(columns=["GradeClass"])
        y_test_app = data["GradeClass"]

        # -------------------------------------------------
        # Select Model and Make Predictions
        # -------------------------------------------------

        if model_choice == "Logistic Regression":

            X_scaled = scaler.transform(X_test_app)

            model = logistic_model

            y_pred = model.predict(X_scaled)

            y_prob = model.predict_proba(X_scaled)

        elif model_choice == "Decision Tree":

            model = tree_model

            y_pred = model.predict(X_test_app)

            y_prob = model.predict_proba(X_test_app)

        elif model_choice == "KNN":

            X_scaled = scaler.transform(X_test_app)

            model = knn_model

            y_pred = model.predict(X_scaled)

            y_prob = model.predict_proba(X_scaled)

        elif model_choice == "Gaussian Naive Bayes":

            X_scaled = scaler.transform(X_test_app)

            model = nb_model

            y_pred = model.predict(X_scaled)

            y_prob = model.predict_proba(X_scaled)

        elif model_choice == "Random Forest":

            model = rf_model

            y_pred = model.predict(X_test_app)

            y_prob = model.predict_proba(X_test_app)


        # -------------------------------------------------
        # Calculate Evaluation Metrics
        # -------------------------------------------------

        accuracy = accuracy_score(y_test_app, y_pred)

        precision = precision_score(
            y_test_app,
            y_pred,
            average="weighted",
            zero_division=0
        )

        recall = recall_score(
            y_test_app,
            y_pred,
            average="weighted",
            zero_division=0
        )

        f1 = f1_score(
            y_test_app,
            y_pred,
            average="weighted",
            zero_division=0
        )

        mcc = matthews_corrcoef(
            y_test_app,
            y_pred
        )

        auc = roc_auc_score(
            y_test_app,
            y_prob,
            multi_class="ovr",
            average="weighted"
        )


        # -------------------------------------------------
        # Display Metrics
        # -------------------------------------------------

        st.subheader("Model Evaluation Results")

        col1, col2, col3 = st.columns(3)

        col1.metric("Accuracy", round(accuracy, 4))
        col2.metric("AUC", round(auc, 4))
        col3.metric("Precision", round(precision, 4))

        col1, col2, col3 = st.columns(3)

        col1.metric("Recall", round(recall, 4))
        col2.metric("F1 Score", round(f1, 4))
        col3.metric("MCC", round(mcc, 4))


        # -------------------------------------------------
        # Confusion Matrix
        # -------------------------------------------------

        st.subheader("Confusion Matrix")

        cm = confusion_matrix(
            y_test_app,
            y_pred
        )

        fig, ax = plt.subplots()

        ConfusionMatrixDisplay(
            confusion_matrix=cm
        ).plot(ax=ax)

        st.pyplot(fig)