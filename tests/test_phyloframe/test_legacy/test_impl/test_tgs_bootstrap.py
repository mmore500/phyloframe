import functools

import numpy as np
import pandas as pd

from phyloframe._auxlib import identity
from phyloframe.legacy._impl import (
    tgs_bootstrap,
    tgs_fit_allmdc_traits,
    tgs_partition_allmdc_branches,
)

from ._simulate_allmdc_branches import simulate_allmdc_branches


def test_replicate():
    branches = simulate_allmdc_branches(4_000)
    fit_kwargs = dict(
        branches=branches,
        partitioning=tgs_partition_allmdc_branches(
            branches=branches,
            neutral_traits=[1, 2],
            trait_names=["focal", "n1", "n2"],
            calibrate=False,
            calibrate_neighborhood_fractions=(1.0,),
            calibrate_num_partitions=64,
            calibrate_size_partition_ratio=4.0,
            calibrate_size_partition_min_neutral_gains=1,
            num_survival_time_bins=8,
        ),
        neutral_traits=[1, 2],
        calibrate=False,
        loss_rate=0.0,
        selection_starts=(0.0, -0.3),
        growth_exponent_bound=600.0,
    )
    kwargs = dict(
        fit=functools.partial(tgs_fit_allmdc_traits, **fit_kwargs),
        num_items=len(branches),
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
