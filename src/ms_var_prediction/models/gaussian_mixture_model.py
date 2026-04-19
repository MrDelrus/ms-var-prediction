import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.base import BaseEstimator, RegressorMixin
from ms_var_prediction.utils import quantile_bisection, gaussian_mixture_cdf
from typing import Any, Optional


class GaussianMixtureVaR(BaseEstimator, RegressorMixin):
    """
    Gaussian Mixture Model for Value-at-Risk (VaR) prediction.

    This model fits a Gaussian mixture to historical returns and estimates
    the VaR at a specified significance level using the mixture's quantile.

    Parameters
    ----------
    n_components : int, default=2
        Number of mixture components.
    covariance_type : str, default='full'
        Type of covariance parameters ('full', 'tied', 'diag', 'spherical').
    tol : float, default=1e-4
        Convergence tolerance for the EM algorithm.
    max_iter : int, default=100
        Maximum number of EM iterations.

    Attributes
    ----------
    gmm_ : GaussianMixture
        Fitted underlying GMM estimator.
    weights_ : np.ndarray
        Mixture weights of each component.
    means_ : np.ndarray
        Means of each component.
    covariances_ : np.ndarray
        Covariances of each component.
    """

    def __init__(
        self,
        n_components: int = 2,
        covariance_type: str = "full",
        tol: float = 1e-4,
        max_iter: int = 100,
    ) -> None:
        if not isinstance(n_components, int) or n_components < 1:
            raise ValueError("n_components must be integer >= 1")
        self.n_components = n_components
        self.covariance_type = covariance_type
        if not isinstance(tol, (int, float)) or tol <= 0:
            raise ValueError("tol must be positive float")
        self.tol = float(tol)
        if not isinstance(max_iter, int) or max_iter < 1:
            raise ValueError("max_iter must be integer >= 1")
        self.max_iter = max_iter

        self.gmm_ = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            tol=self.tol,
            max_iter=self.max_iter,
        )
        self.weights_ = None
        self.means_ = None
        self.covariances_ = None

    def fit(self, returns: np.ndarray, y: Optional[Any] = None) -> "GaussianMixtureVaR":
        """
        Fit the Gaussian mixture model to return data.

        Parameters
        ----------
        returns : array-like, shape (n_samples,)
            Historical returns data.
        y : ignored
            Not used, for compatibility.

        Returns
        -------
        self : object
            Fitted model.
        """
        returns_arr = np.asarray(returns, dtype=np.float64).reshape(-1, 1)
        self.gmm_.fit(returns_arr)
        self.weights_ = self.gmm_.weights_.astype(np.float64)
        self.means_ = self.gmm_.means_.flatten().astype(np.float64)

        if self.covariance_type in ("diag", "spherical"):
            self.covariances_ = np.sqrt(self.gmm_.covariances_.astype(np.float64))
        elif self.covariance_type == "full":
            self.covariances_ = np.sqrt(
                self.gmm_.covariances_.reshape(-1).astype(np.float64)
            )
        else:
            raise ValueError(f"Unsupported covariance_type '{self.covariance_type}'")

        return self

    def get_fitted_params(self) -> dict:
        """Return serialisable dict of all fitted parameters for warm-start / logging."""
        if self.weights_ is None:
            raise ValueError("Model is not fitted yet.")
        return {
            "weights": self.weights_.tolist(),
            "means": self.means_.tolist(),
            "covariances": self.covariances_.tolist(),
            "n_components": self.n_components,
            "covariance_type": self.covariance_type,
        }

    def load_fitted_params(self, params: dict) -> None:
        """Restore fitted state from dict returned by get_fitted_params (skips EM)."""
        self.weights_ = np.asarray(params["weights"], dtype=np.float64)
        self.means_ = np.asarray(params["means"], dtype=np.float64)
        self.covariances_ = np.asarray(params["covariances"], dtype=np.float64)

    def predict(self, var_alpha: float) -> np.float64:
        """
        Estimate the Value-at-Risk (VaR) at a given confidence level.

        Parameters
        ----------
        var_alpha : float
            Significance level (e.g., 0.05 for 5% VaR).

        Returns
        -------
        var : np.float64
            Estimated VaR value.
        """
        return np.float64(
            quantile_bisection(
                gaussian_mixture_cdf,
                var_alpha,
                lower_bound=-1.0,
                upper_bound=1.0,
                means=self.means_,
                std_devs=self.covariances_,
                weights=self.weights_,
                tolerance=1e-4,
            )
        )
