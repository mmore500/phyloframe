import functools

import numpy as np
import pandas as pd

from phyloframe._auxlib import identity
from phyloframe.legacy._impl import (
    tgs_allmdc_fit_traits,
    tgs_allmdc_partition_episodes,
    tgs_bootstrap,
)

from ._simulate_allmdc_exposure_df import simulate_allmdc_exposure_df


def test_replicate():
    exposure_df = simulate_allmdc_exposure_df(4_000)
    fit_kwargs = dict(
        exposure_df=exposure_df,
        partitioning=tgs_allmdc_partition_episodes(
            exposure_df=exposure_df,
            neutral_traits=[1, 2],
            trait_names=["focal", "n1", "n2"],
            calibrate=False,
            calibrate_neighborhood_fractions=(1.0,),
            calibrate_num_partitions=64,
            calibrate_max_rate_ratio=np.inf,
            calibrate_size_partition_ratio=4.0,
            calibrate_size_partition_min_neutral_gains=1,
        ),
        neutral_traits=[1, 2],
        calibrate=False,
        calibrate_max_rate_ratio=np.inf,
        loss_rate=0.0,
        selection_coef_starts=(0.0, -0.3),
        growth_exponent_bound=600.0,
    )
    kwargs = dict(
        fit=functools.partial(tgs_allmdc_fit_traits, **fit_kwargs),
        num_items=len(exposure_df),
        num_bootstrap=2,
        parallel_backend=None,
        progress_wrap=identity,
    )
    fits = tgs_bootstrap(seed=1, **kwargs)
    again = tgs_bootstrap(seed=1, **kwargs)
    other = tgs_bootstrap(seed=2, **kwargs)
    assert len(fits) == 2 * 3
    assert fits.index.names == ["replicate", "trait_index"]
    pd.testing.assert_frame_equal(fits, again)
    assert not np.allclose(fits["gain_rate"], other["gain_rate"])
