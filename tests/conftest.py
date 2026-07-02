import numpy as np
import pytest

from ms_var_prediction.models.gaussian_mixture_model import GaussianMixtureVaR
from ms_var_prediction.models.markov_switching_model import MarkovSwitchingVaR

RNG = np.random.default_rng(42)
N_SAMPLES = 300


@pytest.fixture(scope="session")
def synthetic_returns() -> np.ndarray:
    """Mixture of two Gaussians mimicking regime-switching returns."""
    regime1 = RNG.normal(loc=-0.01, scale=0.02, size=N_SAMPLES // 2)
    regime2 = RNG.normal(loc=0.005, scale=0.008, size=N_SAMPLES // 2)
    returns = np.concatenate([regime1, regime2])
    RNG.shuffle(returns)
    return returns.astype(np.float64)


@pytest.fixture(scope="session")
def fitted_msm(synthetic_returns: np.ndarray) -> MarkovSwitchingVaR:
    model = MarkovSwitchingVaR(n_states=2, optimizer_method="powell")
    model.fit(synthetic_returns)
    return model


@pytest.fixture(scope="session")
def fitted_gmm(synthetic_returns: np.ndarray) -> GaussianMixtureVaR:
    model = GaussianMixtureVaR(n_components=2)
    model.fit(synthetic_returns)
    return model


@pytest.fixture
def clean_exceptions_array() -> np.ndarray:
    """VaR breach pattern: ~5% exceedances, roughly independent."""
    rng = np.random.default_rng(0)
    return rng.binomial(1, 0.05, size=500).astype(int)


@pytest.fixture
def clustered_exceptions_array() -> np.ndarray:
    """VaR breach pattern with clustering (fails independence)."""
    arr = np.zeros(500, dtype=int)
    arr[10:20] = 1
    arr[100:110] = 1
    arr[300:310] = 1
    return arr
