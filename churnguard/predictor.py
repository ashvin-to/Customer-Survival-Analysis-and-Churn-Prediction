"""Core prediction and survival inference engine for ChurnGuard."""

from __future__ import annotations
import base64
import io
from pathlib import Path
import pickle
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge, Rectangle
import numpy as np
import pandas as pd

from churnguard.cltv import compute_cltv


FEATURE_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService",
    "MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "PaperlessBilling", "MonthlyCharges", "TotalCharges",
    "InternetService_Fiber optic", "InternetService_No", "Contract_One year", "Contract_Two year",
    "PaymentMethod_Credit card (automatic)", "PaymentMethod_Electronic check", "PaymentMethod_Mailed check",
]


class ChurnPredictor:
    """Unified inference engine for churn propensity and survival estimation."""

    def __init__(
        self,
        model_path: str | Path = "model.pkl",
        survmodel_path: str | Path = "survivemodel.pkl",
        explainer_path: str | Path = "explainer.bz2",
    ) -> None:
        self.model = pickle.load(open(model_path, "rb"))
        self.survmodel = pickle.load(open(survmodel_path, "rb"))
        self.explainer_path = Path(explainer_path)
        self._explainer = None

    @property
    def explainer(self):
        if self._explainer is None and self.explainer_path.exists():
            try:
                self._explainer = joblib.load(filename=self.explainer_path)
            except Exception:
                self._explainer = None
        return self._explainer

    def extract_features_from_dict(self, form: dict) -> tuple[list, list, int, float]:
        """Parse raw form or API dictionary into formatted numerical vectors."""
        gender = 1 if str(form.get("gender", "0")) == "1" else 0
        SeniorCitizen = 1 if str(form.get("SeniorCitizen", "0")) in ("1", "on", "true") else 0
        Partner = 1 if str(form.get("Partner", "0")) in ("1", "on", "true") else 0
        Dependents = 1 if str(form.get("Dependents", "0")) in ("1", "on", "true") else 0
        PaperlessBilling = 1 if str(form.get("PaperlessBilling", "0")) in ("1", "on", "true") else 0

        MonthlyCharges = float(form.get("MonthlyCharges", 50.0))
        Tenure = int(float(form.get("Tenure", 12)))
        TotalCharges = MonthlyCharges * Tenure

        PhoneService = 1 if str(form.get("PhoneService", "0")) in ("1", "on", "true") else 0
        MultipleLines = 1 if PhoneService == 1 and str(form.get("MultipleLines", "0")) in ("1", "on", "true") else 0

        raw_net = str(form.get("InternetService", "1"))
        InternetService_Fiberoptic = 1 if raw_net == "2" else 0
        InternetService_No = 1 if raw_net == "0" else 0

        OnlineSecurity = 1 if InternetService_No == 0 and str(form.get("OnlineSecurity", "0")) in ("1", "on", "true") else 0
        OnlineBackup = 1 if InternetService_No == 0 and str(form.get("OnlineBackup", "0")) in ("1", "on", "true") else 0
        DeviceProtection = 1 if InternetService_No == 0 and str(form.get("DeviceProtection", "0")) in ("1", "on", "true") else 0
        TechSupport = 1 if InternetService_No == 0 and str(form.get("TechSupport", "0")) in ("1", "on", "true") else 0
        StreamingTV = 1 if InternetService_No == 0 and str(form.get("StreamingTV", "0")) in ("1", "on", "true") else 0
        StreamingMovies = 1 if InternetService_No == 0 and str(form.get("StreamingMovies", "0")) in ("1", "on", "true") else 0

        raw_contract = str(form.get("Contract", "0"))
        Contract_Oneyear = 1 if raw_contract == "1" else 0
        Contract_Twoyear = 1 if raw_contract == "2" else 0

        raw_pay = str(form.get("PaymentMethod", "0"))
        PaymentMethod_CreditCard = 1 if raw_pay == "1" else 0
        PaymentMethod_ElectronicCheck = 1 if raw_pay == "2" else 0
        PaymentMethod_MailedCheck = 1 if raw_pay == "3" else 0

        features = [
            gender, SeniorCitizen, Partner, Dependents, Tenure, PhoneService, MultipleLines,
            OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV,
            StreamingMovies, PaperlessBilling, MonthlyCharges, TotalCharges,
            InternetService_Fiberoptic, InternetService_No, Contract_Oneyear, Contract_Twoyear,
            PaymentMethod_CreditCard, PaymentMethod_ElectronicCheck, PaymentMethod_MailedCheck,
        ]

        surv_feats = [
            gender, SeniorCitizen, Partner, Dependents, PhoneService, MultipleLines,
            OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV,
            StreamingMovies, PaperlessBilling, MonthlyCharges, TotalCharges,
            InternetService_Fiberoptic, InternetService_No, Contract_Oneyear, Contract_Twoyear,
            PaymentMethod_CreditCard, PaymentMethod_ElectronicCheck, PaymentMethod_MailedCheck,
        ]

        return features, surv_feats, Tenure, MonthlyCharges

    def predict(self, form_data: dict) -> dict:
        """Run complete predictive and survival assessment for a customer profile."""
        features, surv_feats, tenure, monthly = self.extract_features_from_dict(form_data)
        
        # Propensity prediction
        prob = float(self.model.predict_proba([features])[0, 1])
        risk_tier = (
            "LOW" if prob < 0.25 else
            "MEDIUM" if prob < 0.50 else
            "HIGH" if prob < 0.75 else
            "EXTREME"
        )

        # Survival curve calculation
        surv_arr = np.array(surv_feats).reshape(1, len(surv_feats))
        surv_series = self.survmodel.predict_survival_function(surv_arr).reset_index()
        surv_series.columns = ["Tenure", "Probability"]
        
        cltv_info = compute_cltv(monthly_charges=monthly, survival_df=surv_series)

        return {
            "churn_probability": round(prob, 4),
            "risk_tier": risk_tier,
            "current_tenure": tenure,
            "monthly_charges": monthly,
            "expected_lifespan_months": cltv_info["expected_lifespan_months"],
            "undiscounted_cltv": cltv_info["undiscounted_cltv"],
            "discounted_npv_cltv": cltv_info["discounted_npv_cltv"],
            "features_vector": features,
            "surv_feats_vector": surv_feats,
            "survival_curve": surv_series,
        }
