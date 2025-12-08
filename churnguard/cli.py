"""ChurnGuard Command-Line Interface."""

from __future__ import annotations
import argparse
from churnguard.predictor import ChurnPredictor


def main() -> None:
    parser = argparse.ArgumentParser(description="ChurnGuard — Customer Survival & Lifetime Value CLI")
    parser.add_argument("--tenure", type=int, default=12, help="Customer tenure in months")
    parser.add_argument("--monthly-charges", type=float, default=65.0, help="Monthly charge in USD")
    parser.add_argument("--fiber", action="store_true", help="Has fiber optic internet")
    parser.add_argument("--contract-two-year", action="store_true", help="Has two-year contract")
    parser.add_argument("--paperless", action="store_true", help="Uses paperless billing")
    args = parser.parse_args()

    predictor = ChurnPredictor()
    sample_input = {
        "Tenure": args.tenure,
        "MonthlyCharges": args.monthly_charges,
        "InternetService": "2" if args.fiber else "1",
        "Contract": "2" if args.contract_two_year else "0",
        "PaperlessBilling": "1" if args.paperless else "0",
    }

    res = predictor.predict(sample_input)

    print("=" * 60)
    print("  🛡️ ChurnGuard Customer Assessment")
    print("=" * 60)
    print(f"  Current Tenure       : {res['current_tenure']} months")
    print(f"  Monthly Charges      : ${res['monthly_charges']:.2f}")
    print(f"  Churn Risk Tier      : [{res['risk_tier']}]")
    print(f"  Churn Probability    : {res['churn_probability'] * 100:.1f}%")
    print(f"  Projected Lifespan   : {res['expected_lifespan_months']:.0f} months")
    print(f"  Undiscounted CLTV    : ${res['undiscounted_cltv']:,.2f}")
    print(f"  Discounted NPV CLTV  : ${res['discounted_npv_cltv']:,.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
