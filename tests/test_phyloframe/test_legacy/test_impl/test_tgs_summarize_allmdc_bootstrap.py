import numpy as np
import pandas as pd

from phyloframe.legacy._impl import tgs_summarize_allmdc_bootstrap


def _res() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trait": ["a", "b"],
            "gain_rate": [2.0, 1.0],
            "selection": [-0.1, 0.0],
            "gain_ratio": [2.0, 0.5],
            "clade_size_shift_2": [10.0, 10.0],
            "clade_size_shift_3": [np.nan, np.nan],
        },
    )


def test_summarize():
    rng = np.random.default_rng(1)
    replicate_fits = [
        (
            pd.DataFrame(
                {
                    "selection": [-0.1 + rng.normal(0, 0.01), rng.normal()],
                    "gain_ratio": [2.0 * np.exp(rng.normal(0, 0.1)), 0.5],
                },
            ),
            np.array([0.0, 10.0 + rng.normal()]),
        )
        for __ in range(20)
    ]
    res = tgs_summarize_allmdc_bootstrap(
        res=_res(),
        replicate_fits=replicate_fits,
        shift_columns=[None, "clade_size_shift_2"],
        num_bootstrap=20,
    )
    assert "selection_z" in res.columns
    assert res["clade_size_shift_2_se"].iloc[0] > 0
    assert res["clade_size_shift_2_se"].nunique() == 1
    # group 3 was not fit, so has no standard error
    assert res["clade_size_shift_3_se"].isna().all()


def test_without_replicates():
    res = tgs_summarize_allmdc_bootstrap(
        res=_res(),
        replicate_fits=[],
        shift_columns=[],
        num_bootstrap=3,
    )
    assert "selection_se" in res.columns
    assert res["clade_size_shift_2_se"].isna().all()
