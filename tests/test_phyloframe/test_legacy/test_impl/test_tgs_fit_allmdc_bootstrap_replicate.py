import numpy as np
import pandas as pd

from phyloframe._auxlib import identity
from phyloframe.legacy._impl import tgs_fit_allmdc_bootstrap_replicate

from ._simulate_allmdc_groups import simulate_allmdc_groups


def test_replicate():
    groups = simulate_allmdc_groups((4_000, 2_000))
    fit_kwargs = dict(
        groups=groups,
        neutral_traits=[1, 2],
        calibrate=False,
        loss_rate=0.0,
        max_shift=100.0,
        num_traits=3,
        starts=None,
        shifts_start=None,
        selection_starts=(0.0, -0.3),
        max_growth_exponent=600.0,
        max_shift_iterations=10,
        progress_wrap=identity,
    )
    fits, shifts = tgs_fit_allmdc_bootstrap_replicate(
        seed=1, fit_kwargs=fit_kwargs
    )
    again, __ = tgs_fit_allmdc_bootstrap_replicate(
        seed=1, fit_kwargs=fit_kwargs
    )
    other, __ = tgs_fit_allmdc_bootstrap_replicate(
        seed=2, fit_kwargs=fit_kwargs
    )
    assert len(fits) == 3 and len(shifts) == 2
    pd.testing.assert_frame_equal(fits, again)
    assert not np.allclose(fits["gain_rate"], other["gain_rate"])
