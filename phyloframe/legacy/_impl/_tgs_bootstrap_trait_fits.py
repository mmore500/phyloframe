import typing

import pandas as pd

from ..._auxlib._run_weighted_bootstrap import run_weighted_bootstrap


def tgs_bootstrap_trait_fits(
    *,
    do_fit: typing.Callable[..., pd.DataFrame],
    num_items: int,
    num_bootstrap: int,
    seed: typing.Optional[int],
    parallel_backend: typing.Optional[str],
    progress_wrap: typing.Callable,
) -> pd.DataFrame:
    """Implementation detail for `alifestd_tgs_pendantmdc_estimate_polars` and
    `alifestd_tgs_allmdc_estimate_polars`. Acronym tgs stands for trait
    gain/selection.

    Returns per-trait fits to bootstrap resamples of `num_items` exposure
    episodes, indexed by replicate and trait index, from `do_fit` (e.g., a
    `functools.partial` of `tgs_fit_traits`), called with each replicate's
    episode `weights`; see `run_weighted_bootstrap`.
    """
    return pd.concat(
        run_weighted_bootstrap(
            do_fit,
            num_items=num_items,
            num_replicates=num_bootstrap,
            seed=seed,
            parallel_backend=parallel_backend,
            progress_wrap=progress_wrap,
        ),
        keys=range(num_bootstrap),  # outer multiindex
        names=["replicate", "trait_index"],
    )
