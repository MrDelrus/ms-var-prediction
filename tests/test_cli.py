import numpy as np
import pandas as pd
import pytest

from ms_var_prediction.cli import _parse_alphas, build_parser, validate
from ms_var_prediction.pipeline import features as fs


def _args(argv):
    return build_parser().parse_args(argv)


# --- alpha parsing --------------------------------------------------------


def test_parse_alphas_ok():
    assert _parse_alphas("0.01,0.05,0.001") == [0.01, 0.05, 0.001]


@pytest.mark.parametrize("bad", ["0.01, 0.05", "0.01;0.05", "abc", "0.05,", "1.5", "0"])
def test_parse_alphas_rejects_bad(bad):
    with pytest.raises(ValueError):
        _parse_alphas(bad)


# --- validation -----------------------------------------------------------


def _base(*extra):
    return ["run", "-m", "GMM", "-b", "2010-01-01", "-e", "2026-01-01", *extra]


def test_validate_ok_features_only(tmp_path):
    validate(_args(_base("-f", str(tmp_path / "f.csv"))))


def test_validate_alphas_requires_results_path(tmp_path):
    with pytest.raises(ValueError, match="results-path is required"):
        validate(_args(_base("-a", "0.05")))


def test_validate_results_path_without_alphas(tmp_path):
    with pytest.raises(ValueError, match="only be used together with --alphas"):
        validate(_args(_base("-r", str(tmp_path / "r.csv"))))


def test_validate_nothing_to_do():
    with pytest.raises(ValueError, match="nothing to do"):
        validate(_args(_base()))


def test_validate_begin_after_end(tmp_path):
    with pytest.raises(ValueError, match="strictly before"):
        validate(
            _args(
                [
                    "run",
                    "-m",
                    "GMM",
                    "-b",
                    "2026-01-01",
                    "-e",
                    "2010-01-01",
                    "-f",
                    str(tmp_path / "f.csv"),
                ]
            )
        )


def test_validate_msm_needs_two_states(tmp_path):
    with pytest.raises(ValueError, match="MSM requires"):
        validate(
            _args(
                [
                    "run",
                    "-m",
                    "MSM",
                    "-n",
                    "1",
                    "-b",
                    "2010-01-01",
                    "-e",
                    "2026-01-01",
                    "-f",
                    str(tmp_path / "f.csv"),
                ]
            )
        )


def test_validate_missing_dir():
    with pytest.raises(ValueError, match="does not exist"):
        validate(_args(_base("-f", "/no/such/dir/f.csv")))


def test_bad_model_rejected_by_argparse():
    with pytest.raises(SystemExit):
        _args(["run", "-m", "XYZ", "-b", "2010-01-01", "-e", "2026-01-01"])


# --- flattened features ---------------------------------------------------


def test_param_columns_msm():
    cols = fs.param_columns("msm", 2)
    assert cols == [
        "mu_0",
        "mu_1",
        "sigma_0",
        "sigma_1",
        "trans_0_0",
        "trans_0_1",
        "trans_1_0",
        "trans_1_1",
        "state_prob_0",
        "state_prob_1",
    ]


def test_param_columns_gmm():
    assert fs.param_columns("gmm", 2) == [
        "weight_0",
        "weight_1",
        "mean_0",
        "mean_1",
        "cov_0",
        "cov_1",
    ]


def _synthetic_gmm_features(n_days=40):
    cols = fs.feature_columns("gmm", 2)
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n_days):
        rows.append(
            {
                "date": f"2015-02-{i % 27 + 1:02d}",
                "ticker": "TST",
                "realized_return": float(rng.normal(0, 0.02)),
                "weight_0": 0.6,
                "weight_1": 0.4,
                "mean_0": 0.001,
                "mean_1": -0.002,
                "cov_0": 0.01,
                "cov_1": 0.03,
            }
        )
    return pd.DataFrame(rows, columns=cols)


def test_predict_and_test_shape():
    df = _synthetic_gmm_features()
    res = fs.predict_and_test(df, "gmm", 2, [0.05, 0.01])
    assert list(res.columns) == ["alpha", "total"] + [
        f"{t}_passed" for t in fs.TEST_NAMES
    ]
    assert len(res) == 2
    assert (res["total"] == 1).all()
    # passed counts are within [0, total]
    for t in fs.TEST_NAMES:
        assert res[f"{t}_passed"].between(0, 1).all()


def test_features_roundtrip_and_verify(tmp_path):
    df = _synthetic_gmm_features()
    path = tmp_path / "features.csv"
    fs.save_features(df, path)
    loaded = fs.load_features(path)
    fs.verify_features(loaded, "gmm", 2, "2010-01-01", "2026-01-01")


def test_verify_features_column_mismatch():
    df = _synthetic_gmm_features()
    with pytest.raises(ValueError, match="columns do not match"):
        fs.verify_features(df, "gmm", 3, "2010-01-01", "2026-01-01")


def test_verify_features_out_of_range():
    df = _synthetic_gmm_features()
    with pytest.raises(ValueError, match="outside the requested range"):
        fs.verify_features(df, "gmm", 2, "2020-01-01", "2021-01-01")
