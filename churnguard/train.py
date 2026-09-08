"""Train and calibrate modern scikit-learn & lifelines model artifacts."""

import pickle
import bz2
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from lifelines import CoxPHFitter
import shap


def prepare_data(filepath="Telco-Customer-Churn.csv"):
    df = pd.read_csv(filepath)
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
        
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].astype(str).str.strip(), errors="coerce").fillna(0)

    for col in ["Partner", "Dependents", "PaperlessBilling", "Churn", "PhoneService"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: 0 if str(x).lower() in ("no", "0") else 1)

    df["gender"] = df["gender"].apply(lambda x: 0 if str(x).lower() == "male" else 1)
    df["MultipleLines"] = df["MultipleLines"].map({"No phone service": 0, "No": 0, "Yes": 1, 0: 0, 1: 1}).fillna(0)

    for col in ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]:
        df[col] = df[col].map({"No internet service": 0, "No": 0, "Yes": 1, 0: 0, 1: 1}).fillna(0)

    df = pd.get_dummies(df, columns=["InternetService", "Contract", "PaymentMethod"], drop_first=True)
    return df


def main():
    print("Preparing dataset...")
    df = prepare_data()

    # Align columns to feature order expected by predictor
    expected_cols = [
        "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService",
        "MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies", "PaperlessBilling", "MonthlyCharges", "TotalCharges",
        "InternetService_Fiber optic", "InternetService_No", "Contract_One year", "Contract_Two year",
        "PaymentMethod_Credit card (automatic)", "PaymentMethod_Electronic check", "PaymentMethod_Mailed check",
    ]

    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0

    X = df[expected_cols]
    y = df["Churn"]

    print("Training modern Random Forest classifier...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
    rf.fit(X, y)
    with open("model.pkl", "wb") as f:
        pickle.dump(rf, f)
    print("Saved model.pkl")

    print("Training Cox Proportional Hazards model...")
    surv_cols = [c for c in expected_cols if c != "tenure"] + ["tenure", "Churn"]
    cph_df = df[surv_cols].copy()
    cph = CoxPHFitter(penalizer=0.01)
    cph.fit(cph_df, duration_col="tenure", event_col="Churn")
    with open("survivemodel.pkl", "wb") as f:
        pickle.dump(cph, f)
    print("Saved survivemodel.pkl")

    print("Calibrating SHAP TreeExplainer...")
    try:
        explainer = shap.TreeExplainer(rf)
        joblib.dump(explainer, filename="explainer.bz2", compress="bz2")
        print("Saved explainer.bz2")
    except Exception as e:
        print(f"SHAP explainer generation note: {e}")

    print("All models successfully retrained and serialized!")


if __name__ == "__main__":
    main()
