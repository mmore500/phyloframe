import functools

import numpy as np
import polars.testing as pltest

from phyloframe._auxlib import identity
from phyloframe.legacy._impl import tgs_fit_traits
from phyloframe.legacy._impl._tgs_bootstrap import _tgs_bootstrap_trait_fits

from ._make_tgs_fit_kwargs import make_tgs_fit_kwargs
from ._simulate_allmdc_exposure_df import simulate_allmdc_exposure_df


def test_replicate():
    exposure_df = simulate_allmdc_exposure_df(4_000)
    kwargs = dict(
        do_fit=functools.partial(
            tgs_fit_traits,
            progress_wrap=identity,
            trait_bin_counts_df=None,
            **make_tgs_fit_kwargs(exposure_df),
        ),
        num_items=len(exposure_df),
        num_bootstrap=2,
        parallel_backend=None,
        progress_wrap=identity,
    )
    fits = _tgs_bootstrap_trait_fits(seed=1, **kwargs)
    again = _tgs_bootstrap_trait_fits(seed=1, **kwargs)
    other = _tgs_bootstrap_trait_fits(seed=2, **kwargs)
    assert len(fits) == 2 * 3
    assert fits["replicate"].to_list() == [0, 0, 0, 1, 1, 1]
    assert fits["trait_id"].to_list() == [0, 1, 2] * 2
    pltest.assert_frame_equal(fits, again)
    assert not np.allclose(fits["gain_rate"], other["gain_rate"])
