import numpy as np
import pandas as pd

from phyloframe.legacy._impl import tgs_build_estimates_df


def _counts() -> pd.DataFrame:
    return pd.DataFrame({"trait": ["a", "b"], "num_leaves": [10, 20]})


def _point() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "gain_rate": [1e-3, 2e-3],
            "selection": [-0.05, 0.0],
            "selection_lrt_naive": [9.0, 0.1],
            "gain_ratio": [0.5, 2.0],
            "gain_ratio_lrt_naive": [3.0, 4.0],
        },
    )


_COLUMNS = [
    "trait",
    "num_leaves",
    "gain_rate",
    "selection",
    "gain_ratio",
    "selection_lrt_naive",
    "gain_ratio_lrt_naive",
]


def test_no_bootstrap():
    out = tgs_build_estimates_df(
        _counts(), point_df=_point(), replicate_fits_df=None, num_bootstrap=0
    )
    assert out.columns.tolist() == _COLUMNS
    assert out["gain_ratio"].tolist() == [0.5, 2.0]


def test_nothing_to_fit():
    out = tgs_build_estimates_df(
        _counts(), point_df=None, replicate_fits_df=None, num_bootstrap=0
    )
    assert out.columns.tolist() == _COLUMNS
    assert (out["gain_rate"] == 0.0).all()
    assert out[_COLUMNS[3:]].isna().all().all()


def test_summary_appended():
    point = _point()
    replicate_fits = pd.concat(
        [
            point.assign(selection=point["selection"] + delta)
            for delta in (-0.01, 0.0, 0.01)
        ],
        keys=range(3),
        names=["replicate", "trait_index"],
    )
    out = tgs_build_estimates_df(
        _counts(),
        point_df=point,
        replicate_fits_df=replicate_fits,
        num_bootstrap=3,
    )
    assert out.columns[: len(_COLUMNS)].tolist() == _COLUMNS
    np.testing.assert_allclose(out["selection_se"], 0.01)


def test_nothing_to_fit_matches_point():
    out = tgs_build_estimates_df(
        _counts(), point_df=_point(), replicate_fits_df=None, num_bootstrap=3
    )
    # identical replicates give zero standard error
    np.testing.assert_allclose(out["selection_se"], 0.0)
    assert (out["num_bootstrap_degenerate"] == 0).all()
