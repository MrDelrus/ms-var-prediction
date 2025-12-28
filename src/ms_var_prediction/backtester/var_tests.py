import numpy as np
import scipy.stats as ss
from typing import Tuple

EPS = 1e-12


def _clip_prob(p: float) -> float:
    """Clip probability to avoid log(0)."""
    return np.clip(p, EPS, 1 - EPS)


def _log_likelihood(k: int, n: int, p: float) -> float:
    p_c = _clip_prob(p)
    return k * np.log(p_c) + (n - k) * np.log(1 - p_c)


def kupiec_test(exceptions: np.ndarray, alpha: float) -> Tuple[float, float]:
    """
    Kupiec unconditional coverage test.
    Returns (LR statistic, p-value)
    """
    n_total = len(exceptions)
    n_exceed = np.sum(exceptions)
    if n_total == 0:
        return np.nan, np.nan

    p_hat = _clip_prob(n_exceed / n_total)
    alpha_c = _clip_prob(alpha)

    ll_actual = _log_likelihood(n_exceed, n_total, p_hat)
    ll_theory = _log_likelihood(n_exceed, n_total, alpha_c)

    lr_stat = -2 * (ll_theory - ll_actual)
    p_val = 1 - ss.chi2.cdf(lr_stat, df=1)
    return lr_stat, p_val


def build_transition_matrix(exceptions: np.ndarray) -> np.ndarray:
    """Build 2x2 transition matrix for independence tests."""
    n = len(exceptions)
    if n <= 1:
        return np.zeros((2, 2), dtype=int)

    shifted = np.roll(exceptions, 1)
    shifted[0] = 0
    matrix = np.zeros((2, 2), dtype=int)
    for prev, cur in zip(shifted, exceptions):
        matrix[prev, cur] += 1
    return matrix


def christoffersen_independence_test(exceptions: np.ndarray) -> Tuple[float, float]:
    """
    Christoffersen independence test.
    Returns (LR statistic, p-value)
    """
    matrix = build_transition_matrix(exceptions)
    n00, n01 = matrix[0]
    n10, n11 = matrix[1]

    n0, n1 = n00 + n01, n10 + n11
    p0 = _clip_prob(n01 / n0) if n0 else 0.5
    p1 = _clip_prob(n11 / n1) if n1 else 0.5

    ll_2state = _log_likelihood(n01, n0, p0) + _log_likelihood(n11, n1, p1)
    total_ones = n01 + n11
    total_n = n0 + n1
    p_hat = _clip_prob(total_ones / total_n)
    ll_pooled = _log_likelihood(total_ones, total_n, p_hat)

    lr_stat = -2 * (ll_pooled - ll_2state)
    p_val = 1 - ss.chi2.cdf(lr_stat, df=1)
    return lr_stat, p_val


def christoffersen_test(exceptions: np.ndarray, alpha: float) -> Tuple[float, float]:
    """
    Christoffersen conditional coverage test (unconditional coverage + independence).
    Returns (LR statistic, p-value)
    """
    lr_uc, _ = kupiec_test(exceptions, alpha)
    lr_ind, _ = christoffersen_independence_test(exceptions)

    lr_cc = lr_uc + lr_ind

    p_val = 1 - ss.chi2.cdf(lr_cc, df=2)
    return lr_cc, p_val
