from __future__ import annotations

from typing import Any

from ms_var_prediction.models.gaussian_mixture_model import GaussianMixtureVaR
from ms_var_prediction.models.markov_switching_model import MarkovSwitchingVaR

__all__ = ["MarkovSwitchingVaR", "GaussianMixtureVaR", "build_model"]


def build_model(model_type: str, hyperparams: dict[str, Any]) -> Any:
    """Construct a VaR model by name. ``model_type`` is 'msm' or 'gmm'."""
    if model_type == "msm":
        return MarkovSwitchingVaR(**hyperparams)
    if model_type == "gmm":
        return GaussianMixtureVaR(**hyperparams)
    raise ValueError(f"Unknown model_type: {model_type!r}")
