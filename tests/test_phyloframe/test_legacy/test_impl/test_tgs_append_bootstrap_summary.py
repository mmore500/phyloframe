import numpy as np
import pandas as pd

from phyloframe.legacy._impl import tgs_append_bootstrap_summary


def _res() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trait": ["a", "b"],
            "gain_rate": [1e-3, 2e-3],
            "selection": [-0.05, 0.0],
            "gain_ratio": [0.5, 2.0],
        },
    )


def test_no_bootstrap():
    res = _res()
    assert (
        tgs_append_bootstrap_summary(res, replicate_fits=None, num_bootstrap=0)
        is res
    )


def test_summary_appended():
    res = _res()
    replicate_fits = pd.concat(
        [
            res.assign(selection=res["selection"] + delta)
            for delta in (-0.01, 0.0, 0.01)
        ],
        keys=range(3),
        names=["replicate", "trait_index"],
    )
    out = tgs_append_bootstrap_summary(
        res, replicate_fits=replicate_fits, num_bootstrap=3
    )
    assert out.columns[: len(res.columns)].tolist() == res.columns.tolist()
    np.testing.assert_allclose(out["selection_se"], 0.01)


def test_nothing_to_fit_matches_point():
    out = tgs_append_bootstrap_summary(
        _res(), replicate_fits=None, num_bootstrap=3
    )
    # identical replicates give zero standard error
    np.testing.assert_allclose(out["selection_se"], 0.0)
    assert (out["num_bootstrap_degenerate"] == 0).all()
