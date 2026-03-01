# 🛡️ ChurnGuard — Customer Survival Analysis & Dynamic CLTV Suite

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Flask-black.svg)](https://flask.palletsprojects.com/)
[![Explainability](https://img.shields.io/badge/XAI-SHAP-brightgreen.svg)](https://shap.readthedocs.io/)
[![Survival Analysis](https://img.shields.io/badge/Survival-Lifelines-blueviolet.svg)](https://lifelines.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE.md)

**ChurnGuard** is a customer retention intelligence suite uniting **Random Forest propensity scoring**, **Cox Proportional Hazards survival analysis**, and **SHAP explainability** to quantify defection risks and dynamically estimate Customer Lifetime Value (CLTV).

---

## 🌟 Key Highlights & Innovations

1. **Dual Modeling Architecture**:
   - **Propensity Model**: Calibrated Random Forest predicting instantaneous churn probability $P(\text{Churn} \mid X)$.
   - **Survival & Hazard Model**: Semi-parametric Cox Proportional Hazards model estimating cumulative hazard $H(t \mid X)$ and survival probability curves $S(t \mid X) = \exp(-H(t \mid X))$ across tenure horizons.

2. **Dynamic & Discounted CLTV Engine**:
   - Computes expected customer lifespan from survival curve quantiles ($S(t) \ge 0.10$).
   - Calculates both undiscounted gross lifetime value and Net Present Value (NPV) discounted cash flow.

3. **Game-Theoretic XAI (SHAP)**:
   - Evaluates exact feature contributions (TreeExplainer) explaining individual customer risk drivers.

4. **Modular Architecture & REST API**:
   - Clean, decoupled `churnguard` Python package.
   - Production REST API endpoint (`/api/v1/predict`) for headless ingestion and batch pipelines.
   - Standalone CLI for single-record evaluations.

---

## 📂 Repository Structure

```text
ChurnGuard/
├── churnguard/                         <- Core modular Python library
│   ├── __init__.py
│   ├── predictor.py                    <- ChurnPredictor inference engine
│   ├── cltv.py                         <- Dynamic & NPV CLTV calculation utilities
│   └── cli.py                          <- CLI evaluation tool
├── templates/
│   └── index.html                      <- Web dashboard template
├── static/                             <- Static web assets and figures
├── Customer Survival Analysis.ipynb    <- Kaplan-Meier & Cox Proportional Hazards
├── Exploratory Data Analysis.ipynb     <- Demographic and tenure data analysis
├── Churn Prediction Model.ipynb        <- Random Forest classifier training notebook
├── app.py                              <- Flask Web dashboard & REST API
├── explainer.bz2                       <- SHAP TreeExplainer artifact
├── model.pkl                           <- Calibrated classifier artifact
├── survivemodel.pkl                    <- Calibrated survival model artifact
├── requirements.txt
├── Procfile
├── LICENSE.md
└── README.md
```

---

## 🚀 Quick Start

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Command Line Interface (CLI)
```bash
python -m churnguard.cli --tenure 18 --monthly-charges 75.0 --fiber --paperless
```

Sample output:
```text
============================================================
  🛡️ ChurnGuard Customer Assessment
============================================================
  Current Tenure       : 18 months
  Monthly Charges      : $75.00
  Churn Risk Tier      : [MEDIUM]
  Churn Probability    : 42.1%
  Projected Lifespan   : 54 months
  Undiscounted CLTV    : $4,050.00
  Discounted NPV CLTV  : $3,512.80
============================================================
```

### 3. Launch Web Dashboard & REST API
```bash
python app.py
```
- Web UI: `http://localhost:5000`
- REST Endpoint: `POST http://localhost:5000/api/v1/predict` (JSON payload)

---

## 📐 Mathematical Formulation

### 1. Cox Proportional Hazards
The hazard rate for customer $i$ at tenure $t$:
$$h(t \mid X_i) = h_0(t) \exp\left(\sum_{j=1}^p \beta_j X_{i,j}\right)$$

### 2. Survival Probability
$$S(t \mid X_i) = \exp\left(-\int_0^t h(u \mid X_i) \, du\right)$$

### 3. Discounted Customer Lifetime Value (NPV)
$$\text{CLTV}_{\text{NPV}} = \sum_{t=1}^{T_{\text{max}}} \frac{\text{MRR}_t}{(1 + r_{\text{monthly}})^t}$$
where $T_{\text{max}} = \max \{ t \mid S(t) \ge 0.10 \}$.

---

## 📜 License
MIT License.
