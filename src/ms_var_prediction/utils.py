import numpy as np
import pandas as pd
from math import exp, sqrt, pi, erf
from numba import njit
from typing import Callable
from ms_var_prediction.constants import SIGMA_EPS


def prices_to_returns(prices: pd.Series) -> np.ndarray:
    """Convert price series to log returns."""
    return np.log(prices / prices.shift(1)).dropna().values


@njit
def quantile_bisection(
    cdf_function: Callable[[float, np.ndarray, np.ndarray, np.ndarray], np.float64],
    alpha: float,
    lower_bound: float,
    upper_bound: float,
    means: np.ndarray,
    std_devs: np.ndarray,
    weights: np.ndarray,
    tolerance: float = 1e-4,
) -> np.float64:
    """
    Compute the alpha-quantile of a monotonic CDF using the bisection method.

    Parameters
    ----------
    cdf_function : callable
        Function of the form f(x, means, std_devs, weights) returning a CDF value.
    alpha : float
        Target CDF value (e.g., 0.05 for VaR).
    lower_bound : float
        Lower bound of search interval.
    upper_bound : float
        Upper bound of search interval.
    means : np.ndarray
        Means of mixture components.
    std_devs : np.ndarray
        Standard deviations of mixture components.
    weights : np.ndarray
        Weights of mixture components.
    tolerance : float
        Convergence tolerance.

    Returns
    -------
    root : np.float64
        The alpha-quantile of the distribution.
    """
    lower = lower_bound
    upper = upper_bound
    while (upper - lower) > tolerance:
        midpoint = (lower + upper) / 2.0
        if cdf_function(midpoint, means, std_devs, weights) < alpha:
            lower = midpoint
        else:
            upper = midpoint
    return np.float64((lower + upper) / 2.0)


@njit
def gaussian_cdf(x: float, mean: float, std: float) -> np.float64:
    """
    Compute the CDF of a Gaussian distribution.
    """
    std = max(abs(std), SIGMA_EPS)
    z = (x - mean) / std
    return np.float64(0.5 * (1.0 + erf(z / sqrt(2.0))))


@njit
def gaussian_pdf(x: float, mean: float, std: float) -> np.float64:
    """
    Compute the PDF of a Gaussian distribution.
    """
    std = max(abs(std), SIGMA_EPS)
    u = (x - mean) / abs(std)
    y = exp(-0.5 * u * u) / (sqrt(2.0 * pi) * abs(std))
    return np.float64(y)


@njit
def gaussian_mixture_cdf(
    x: float, means: np.ndarray, std_devs: np.ndarray, weights: np.ndarray
) -> np.float64:
    """
    Compute the CDF of a Gaussian mixture distribution.
    """
    result = np.float64(0.0)
    for i in range(len(means)):
        result += weights[i] * gaussian_cdf(x, means[i], std_devs[i])
    return result
