import numpy as np
import pandas as pd

from phyloframe.legacy._impl import (
    tgs_fit_allmdc_bootstrap_replicate,
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
        max_growth_exponent=600.0,
    )
    fits = tgs_fit_allmdc_bootstrap_replicate(seed=1, fit_kwargs=fit_kwargs)
    again = tgs_fit_allmdc_bootstrap_replicate(seed=1, fit_kwargs=fit_kwargs)
    other = tgs_fit_allmdc_bootstrap_replicate(seed=2, fit_kwargs=fit_kwargs)
    assert len(fits) == 3
    pd.testing.assert_frame_equal(fits, again)
    assert not np.allclose(fits["gain_rate"], other["gain_rate"])
