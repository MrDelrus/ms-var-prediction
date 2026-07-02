import numpy as np

from ms_var_prediction.backtester.var_tests import (
    build_transition_matrix,
    christoffersen_independence_test,
    christoffersen_test,
    kupiec_test,
)


class TestKupiecTest:
    def test_returns_tuple(self, clean_exceptions_array):
        result = kupiec_test(clean_exceptions_array, alpha=0.05)
        assert len(result) == 2

    def test_correct_frequency_passes(self, clean_exceptions_array):
        _, pval = kupiec_test(clean_exceptions_array, alpha=0.05)
        assert pval > 0.05

    def test_wrong_frequency_fails(self):
        # ~20% breach rate vs 5% expected → should fail
        exceptions = np.zeros(500, dtype=int)
        exceptions[:100] = 1
        _, pval = kupiec_test(exceptions, alpha=0.05)
        assert pval < 0.05

    def test_empty_input(self):
        lr, pval = kupiec_test(np.array([]), alpha=0.05)
        assert np.isnan(lr) and np.isnan(pval)

    def test_lr_stat_nonnegative(self, clean_exceptions_array):
        lr, _ = kupiec_test(clean_exceptions_array, alpha=0.05)
        assert lr >= 0


class TestChristoffersenIndependenceTest:
    def test_returns_tuple(self, clean_exceptions_array):
        result = christoffersen_independence_test(clean_exceptions_array)
        assert len(result) == 2

    def test_independent_breaches_pass(self, clean_exceptions_array):
        _, pval = christoffersen_independence_test(clean_exceptions_array)
        assert pval > 0.05

    def test_clustered_breaches_fail(self, clustered_exceptions_array):
        _, pval = christoffersen_independence_test(clustered_exceptions_array)
        assert pval < 0.05

    def test_pval_in_range(self, clean_exceptions_array):
        _, pval = christoffersen_independence_test(clean_exceptions_array)
        assert 0.0 <= pval <= 1.0


class TestChristoffersenConditionalTest:
    def test_returns_tuple(self, clean_exceptions_array):
        result = christoffersen_test(clean_exceptions_array, alpha=0.05)
        assert len(result) == 2

    def test_good_series_passes(self, clean_exceptions_array):
        _, pval = christoffersen_test(clean_exceptions_array, alpha=0.05)
        assert pval > 0.05

    def test_lr_stat_equals_sum_of_components(self, clean_exceptions_array):
        lr_uc, _ = kupiec_test(clean_exceptions_array, alpha=0.05)
        lr_ind, _ = christoffersen_independence_test(clean_exceptions_array)
        lr_cc, _ = christoffersen_test(clean_exceptions_array, alpha=0.05)
        assert np.isclose(lr_cc, lr_uc + lr_ind)


class TestBuildTransitionMatrix:
    def test_shape(self, clean_exceptions_array):
        m = build_transition_matrix(clean_exceptions_array)
        assert m.shape == (2, 2)

    def test_sum_equals_length(self, clean_exceptions_array):
        # implementation sets shifted[0]=0, producing n pairs total (not n-1)
        m = build_transition_matrix(clean_exceptions_array)
        assert m.sum() == len(clean_exceptions_array)

    def test_all_zeros(self):
        arr = np.zeros(10, dtype=int)
        m = build_transition_matrix(arr)
        assert m[0, 0] == 10
        assert m[0, 1] == 0

    def test_short_array(self):
        m = build_transition_matrix(np.array([1]))
        assert m.sum() == 0
