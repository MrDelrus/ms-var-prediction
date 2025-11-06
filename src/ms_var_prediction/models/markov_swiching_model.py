import numpy as np
from scipy import optimize
from sklearn.base import BaseEstimator, RegressorMixin
from ms_var_prediction.utils import quantile_bisection, gaussian_mixture_cdf
from ms_var_prediction.models.loglikelyhood import loglikelyhood_gaussian
from typing import Optional, Dict

_ALLOWED_OPTIM_METHODS = ["powell", "BFGS", "L-BFGS-B", "CG", "Nelder-Mead", "TNC"]


class MarkovSwitchingVaR(BaseEstimator, RegressorMixin):
    """
    N-state Gaussian Markov Switching Model for VaR prediction.

    Parameters
    ----------
    n_states : int, default=2
        Number of hidden states (>=2).
    epsilon : float, default=1e-4
        Convergence tolerance for optimizer.
    init_params : np.ndarray, optional
        Initial parameters [mus..., sigmas..., P_flat]. If None, initialized automatically.
    optimizer_method : str, default='powell'
        Optimization method. Must be one of _ALLOWED_OPTIM_METHODS.
    optimizer_options : dict, optional
        Additional options for the optimizer (e.g., {'maxiter':50}).

    Attributes
    ----------
    _params : np.ndarray
        Flattened array of [mus, sigmas, P_flat].
    _state_probs : np.ndarray
        Probabilities of each state at the last observation.
    """

    def __init__(
        self,
        n_states: int = 2,
        epsilon: float = 1e-4,
        init_params: Optional[np.ndarray] = None,
        optimizer_method: str = "powell",
        optimizer_options: Optional[Dict[str, float]] = None,
    ) -> None:
        if not isinstance(n_states, int) or n_states < 2:
            raise ValueError("n_states must be an integer >= 2")
        self.n_states = n_states

        if not isinstance(epsilon, (int, float)) or epsilon <= 0:
            raise ValueError("epsilon must be a positive float")
        self.epsilon = float(epsilon)

        if optimizer_method not in _ALLOWED_OPTIM_METHODS:
            raise ValueError(
                f"optimizer_method '{optimizer_method}' not valid. "
                f"Choose one of {_ALLOWED_OPTIM_METHODS}"
            )
        self.optimizer_method = optimizer_method

        default_options = {"maxiter": 50, "disp": 0.0}
        if optimizer_options is not None:
            default_options.update(optimizer_options)
        self.optimizer_options = default_options

        if init_params is not None:
            init_params = np.asarray(init_params, dtype=np.float64)
            expected_len = n_states * (2 + n_states)
            if init_params.size != expected_len:
                raise ValueError(f"init_params must have length {expected_len}")
        self._params = init_params
        self._state_probs = None

    def fit(self, returns, y=None):
        """
        Fit the MS model to returns.

        Parameters
        ----------
        returns : array-like, shape (n_samples,)
            Time series of returns.
        y : Ignored
            Sklearn compatibility.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        returns_arr = np.asarray(returns, dtype=np.float64)
        mu0 = returns_arr.mean()
        sigma0 = returns_arr.std()
        n = self.n_states

        if self._params is None:
            mus = np.linspace(mu0 - sigma0, mu0 + sigma0, n, dtype=np.float64)
            sigmas = np.full(n, sigma0, dtype=np.float64)
            P_flat = np.full(n * n, 1.0 / n, dtype=np.float64)
            self._params = np.hstack([mus, sigmas, P_flat])

        bounds = (
            [(mu0 - 3 * sigma0, mu0 + 3 * sigma0)] * n
            + [(sigma0 / 10, 3 * sigma0)] * n
            + [(0.0, 1.0)] * (n * n)
        )

        result = optimize.minimize(
            lambda x: loglikelyhood_gaussian(x.astype(np.float64), returns_arr, n)[0],
            self._params,
            method=self.optimizer_method,
            tol=self.epsilon,
            bounds=bounds,
            options=self.optimizer_options,
        )

        self._params = result.x.astype(np.float64)
        _, self._state_probs = loglikelyhood_gaussian(self._params, returns_arr, n)
        return self

    def predict(self, var_alpha):
        """
        Compute Value-at-Risk at a given alpha.

        Parameters
        ----------
        var_alpha : float
            Significance level (e.g., 0.05 for 5% VaR).

        Returns
        -------
        var : np.float64
            Estimated VaR.
        """
        n = self.n_states
        mus = self._params[:n].astype(np.float64)
        sigmas = self._params[n : 2 * n].astype(np.float64)
        weights = self._state_probs.astype(np.float64)
        return np.float64(
            quantile_bisection(
                gaussian_mixture_cdf, var_alpha, -1, 1, mus, sigmas, weights
            )
        )
