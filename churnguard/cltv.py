"""Customer Lifetime Value (CLTV) calculation utilities."""

from __future__ import annotations
import numpy as np
import pandas as pd


def compute_cltv(
    monthly_charges: float,
    survival_df: pd.DataFrame,
    survival_threshold: float = 0.10,
    annual_discount_rate: float = 0.08,
) -> dict[str, float]:
    """Compute undiscounted and discounted dynamic Customer Lifetime Value.

    Args:
        monthly_charges: Monthly recurring revenue from customer.
        survival_df: DataFrame with columns ['Tenure', 'Probability'].
        survival_threshold: Cutoff survival probability to define maximum lifespan.
        annual_discount_rate: Annual financial discount rate for NPV.

    Returns:
        Dictionary containing max lifespan, undiscounted CLTV, and NPV discounted CLTV.
    """
    viable_tenures = survival_df[survival_df["Probability"] >= survival_threshold]
    max_months = float(viable_tenures["Tenure"].max()) if not viable_tenures.empty else 0.0

    undiscounted_cltv = round(max_months * monthly_charges, 2)

    # Net Present Value discounted CLTV
    monthly_discount = (1.0 + annual_discount_rate) ** (1.0 / 12.0) - 1.0
    discount_factors = [1.0 / ((1.0 + monthly_discount) ** t) for t in range(1, int(max_months) + 1)]
    npv_cltv = round(float(sum(monthly_charges * df for df in discount_factors)), 2)

    return {
        "expected_lifespan_months": max_months,
        "undiscounted_cltv": undiscounted_cltv,
        "discounted_npv_cltv": npv_cltv,
    }
