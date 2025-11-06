import numpy as np
import scipy.stats as ss

from ms_var_prediction.evaluation.var_metrics import (
    kupiec_test,
    christoffersen_independence_test,
    christoffersen_test,
)


class Backtester:
    """
    Backtester for VaR predictions.

    Computes multiple statistical tests:
    - Binomial test (unconditional coverage)
    - Independence test
    - Kupiec test
    - Christoffersen independence test
    - Christoffersen conditional coverage test
    """

    def __init__(self, returns: np.ndarray, vars: np.ndarray, alpha: float):
        self._returns = np.array(returns)
        self._vars = np.array(vars)
        self._alpha = alpha
        self._n = self._returns.shape[0]

    def _compute_exceptions(self) -> np.ndarray:
        """Binary array indicating VaR exceedances."""
        return (self._returns < self._vars).astype(int)

    def _binomial_test(self, exceptions: np.ndarray) -> bool:
        """Unconditional coverage via binomial test."""
        pval = ss.binomtest(exceptions.sum(), n=self._n, p=self._alpha).pvalue
        return pval > 0.05

    def _independence_test(self, exceptions: np.ndarray) -> bool:
        """Simple independence test using transition counts."""
        shifted = np.roll(exceptions, 1)
        shifted[0] = 0
        matrix = np.zeros((2, 2), dtype=int)
        for prev, curr in zip(shifted, exceptions):
            matrix[prev, curr] += 1

        n00, n01 = matrix[0]
        n10, n11 = matrix[1]

        numerator = np.abs(n00 * n11 - n01 * n10) - self._n / 2
        denominator = (n00 + n01) * (n00 + n10) * (n11 + n01) * (n11 + n10)
        indep_stat = self._n * numerator**2 / denominator if denominator != 0 else 0
        return indep_stat < ss.chi2.ppf(0.95, df=1)

    def test(self) -> dict:
        """
        Run all backtests.
        Returns a dictionary with True/False for each test.
        """
        exceptions = self._compute_exceptions()

        results = {
            "binomial": self._binomial_test(exceptions),
            "independence_simple": self._independence_test(exceptions),
            "kupiec": kupiec_test(exceptions, self._alpha)[1] > 0.05,
            "christoffersen_independence": christoffersen_independence_test(exceptions)[
                1
            ]
            > 0.05,
            "christoffersen_conditional": christoffersen_test(exceptions, self._alpha)[
                1
            ]
            > 0.05,
        }

        return results
