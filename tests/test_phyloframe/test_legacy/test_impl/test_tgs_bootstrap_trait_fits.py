import functools

import numpy as np
import pandas as pd

from phyloframe._auxlib import identity
from phyloframe.legacy._impl import tgs_bootstrap_trait_fits, tgs_fit_traits

from ._make_tgs_fit_kwargs import make_tgs_fit_kwargs
from ._simulate_allmdc_exposure_df import simulate_allmdc_exposure_df


def test_replicate():
    exposure_df = simulate_allmdc_exposure_df(4_000)
    kwargs = dict(
        do_fit=functools.partial(
            tgs_fit_traits, **make_tgs_fit_kwargs(exposure_df)
        ),
        num_items=len(exposure_df),
        num_bootstrap=2,
        parallel_backend=None,
        progress_wrap=identity,
    )
    fits = tgs_bootstrap_trait_fits(seed=1, **kwargs)
    again = tgs_bootstrap_trait_fits(seed=1, **kwargs)
    other = tgs_bootstrap_trait_fits(seed=2, **kwargs)
    assert len(fits) == 2 * 3
    assert fits.index.names == ["replicate", "trait_index"]
    pd.testing.assert_frame_equal(fits, again)
    assert not np.allclose(fits["gain_rate"], other["gain_rate"])
