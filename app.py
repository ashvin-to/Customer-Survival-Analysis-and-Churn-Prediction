"""ChurnGuard Web Dashboard and API Server."""

from __future__ import annotations
import base64
import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge, Rectangle
import numpy as np
import shap
shap.initjs()
from flask import Flask, jsonify, render_template, request

from churnguard.predictor import ChurnPredictor, FEATURE_COLUMNS

app = Flask(__name__)
engine = ChurnPredictor()


def create_gauge(probability: float) -> str:
    """Generate a clean speedometer gauge chart for churn risk."""
    gauge_img = io.BytesIO()
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # 4 risk segments
    starts = np.array([0, 45, 90, 135])
    ends = np.array([45, 90, 135, 180])
    mids = (starts + ends) / 2.0
    colors = ["#ED1C24", "#FFCC00", "#0063BF", "#007A00"]
    labels = ["EXTREME", "HIGH", "MEDIUM", "LOW"]

    for s, e, c in zip(starts, ends, colors):
        ax.add_patch(Wedge((0.0, 0.0), 0.4, s, e, facecolor="w", lw=1.5))
        ax.add_patch(Wedge((0.0, 0.0), 0.4, s, e, width=0.10, facecolor=c, lw=1.5, alpha=0.6))

    for mid, lab in zip(mids, labels):
        ang_rad = np.radians(mid)
        rot = np.degrees(ang_rad) - 90
        ax.text(
            0.35 * np.cos(ang_rad),
            0.35 * np.sin(ang_rad),
            lab,
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=10,
            fontweight="bold",
            rotation=rot,
        )

    r = Rectangle((-0.4, -0.1), 0.8, 0.1, facecolor="w", lw=1.5)
    ax.add_patch(r)
    ax.text(
        0,
        -0.05,
        f"Churn Risk: {probability * 100:.1f}%",
        horizontalalignment="center",
        verticalalignment="center",
        fontsize=14,
        fontweight="bold",
    )

    pos = (1.0 - probability) * 180.0
    pos_rad = np.radians(pos)
    ax.arrow(
        0,
        0,
        0.225 * np.cos(pos_rad),
        0.225 * np.sin(pos_rad),
        width=0.03,
        head_width=0.08,
        head_length=0.08,
        fc="k",
        ec="k",
    )
    ax.add_patch(Circle((0, 0), radius=0.02, facecolor="k"))

    ax.set_frame_on(False)
    ax.axes.set_xticks([])
    ax.axes.set_yticks([])
    ax.axis("equal")
    plt.tight_layout()

    plt.savefig(gauge_img, format="png")
    plt.close(fig)
    gauge_img.seek(0)
    return base64.b64encode(gauge_img.getvalue()).decode()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/v1/predict", methods=["POST"])
def api_predict():
    data = request.get_json(force=True) if request.is_json else request.form.to_dict()
    res = engine.predict(data)
    return jsonify({
        "status": "success",
        "data": {
            "churn_probability": res["churn_probability"],
            "risk_tier": res["risk_tier"],
            "current_tenure_months": res["current_tenure"],
            "expected_lifespan_months": res["expected_lifespan_months"],
            "undiscounted_cltv_usd": res["undiscounted_cltv"],
            "discounted_npv_cltv_usd": res["discounted_npv_cltv"],
        },
    })


@app.route("/predict", methods=["POST"])
def predict():
    res = engine.predict(request.form)
    
    # SHAP explanations
    shap_url = ""
    if engine.explainer is not None:
        try:
            shap_values = engine.explainer.shap_values(np.array([res["features_vector"]]))
            shap_img = io.BytesIO()
            shap.force_plot(
                engine.explainer.expected_value[1],
                shap_values[1],
                FEATURE_COLUMNS,
                matplotlib=True,
                show=False,
            ).savefig(shap_img, bbox_inches="tight", format="png")
            shap_img.seek(0)
            shap_url = base64.b64encode(shap_img.getvalue()).decode()
            plt.close("all")
        except Exception:
            pass

    # Hazard Curve
    surv_arr = np.array(res["surv_feats_vector"]).reshape(1, len(res["surv_feats_vector"]))
    hazard_img = io.BytesIO()
    fig, ax = plt.subplots(figsize=(6, 4))
    engine.survmodel.predict_cumulative_hazard(surv_arr).plot(ax=ax, color="#d62728", lw=2)
    ax.axvline(x=res["current_tenure"], color="#1f77b4", linestyle="--", label=f"Current Tenure ({res['current_tenure']}m)")
    ax.set_xlabel("Tenure (Months)")
    ax.set_ylabel("Cumulative Hazard")
    ax.set_title("Projected Cumulative Hazard Rate")
    ax.legend()
    plt.tight_layout()
    plt.savefig(hazard_img, format="png")
    plt.close(fig)
    hazard_img.seek(0)
    hazard_url = base64.b64encode(hazard_img.getvalue()).decode()

    # Survival Curve
    surv_img = io.BytesIO()
    fig, ax = plt.subplots(figsize=(6, 4))
    engine.survmodel.predict_survival_function(surv_arr).plot(ax=ax, color="#2ca02c", lw=2)
    ax.axvline(x=res["current_tenure"], color="#1f77b4", linestyle="--", label=f"Current Tenure ({res['current_tenure']}m)")
    ax.set_xlabel("Tenure (Months)")
    ax.set_ylabel("Survival Probability")
    ax.set_title("Survival Probability Over Lifetime")
    ax.legend()
    plt.tight_layout()
    plt.savefig(surv_img, format="png")
    plt.close(fig)
    surv_img.seek(0)
    surv_url = base64.b64encode(surv_img.getvalue()).decode()

    gauge_url = create_gauge(res["churn_probability"])

    return render_template(
        "index.html",
        prediction_text=f"Risk Tier: [{res['risk_tier']}] | Churn Prob: {res['churn_probability']*100:.1f}% | Expected CLTV: ${res['undiscounted_cltv']:,.2f} (NPV: ${res['discounted_npv_cltv']:,.2f})",
        url_1=gauge_url,
        url_2=shap_url,
        url_3=hazard_url,
        url_4=surv_url,
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
