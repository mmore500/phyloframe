import numpy as np
import polars as pl

from phyloframe._auxlib import identity
from phyloframe.legacy._impl import tgs_bootstrap_tgs_from_exposures

from ._simulate_allmdc_exposure_df import simulate_allmdc_exposure_df

_TRAIT_NAMES = ["focal", "a", "b", "c"]


def _bootstrap(exposure_df: pl.DataFrame, **kwargs) -> pl.DataFrame:
    return tgs_bootstrap_tgs_from_exposures(
        exposure_df,
        **{
            **dict(
                trait_names=_TRAIT_NAMES,
                neutral_traits=[1, 2, 3],
                calibrate_neighborhood_fractions=(1.0,),
                calibrate_max_rate_ratio=np.inf,
                length_rtol=1e-2,
                loss_rate=0.0,
                selection_coef_starts=(0.0, -0.3),
                growth_exponent_bound=600.0,
                num_bootstrap=3,
                seed=1,
                parallel_backend=None,
                progress_wrap=identity,
            ),
            **kwargs,
        },
    )


def test_replicates():
    exposure_df = simulate_allmdc_exposure_df(4_000, num_neutral=3)
    res = _bootstrap(exposure_df)
    assert res.columns == [
        "replicate",
        "trait",
        "gain_rate",
        "selection_coef",
        "selection_coef_lrt_naive",
        "gain_ratio",
        "gain_ratio_lrt_naive",
    ]
    assert res["replicate"].to_list() == [
        r for r in range(3) for __ in range(4)
    ]
    assert res["trait"].to_list() == _TRAIT_NAMES * 3
    assert res.equals(_bootstrap(exposure_df))
    assert not res.equals(_bootstrap(exposure_df, seed=2))
