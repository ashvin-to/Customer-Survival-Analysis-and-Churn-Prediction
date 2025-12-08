"""ChurnGuard: Customer Survival Analytics & Lifetime Value Estimation."""

from churnguard.predictor import ChurnPredictor
from churnguard.cltv import compute_cltv

__all__ = ["ChurnPredictor", "compute_cltv"]
__version__ = "1.0.0"
