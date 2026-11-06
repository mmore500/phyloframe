import typing

import polars as pl

from ..._auxlib._run_weighted_bootstrap import run_weighted_bootstrap


def tgs_bootstrap_trait_fits(
    *,
    do_fit: typing.Callable[..., pl.DataFrame],
    num_items: int,
    num_bootstrap: int,
    seed: typing.Optional[int],
    parallel_backend: typing.Optional[str],
    progress_wrap: typing.Callable,
) -> pl.DataFrame:
    """Implementation detail for `alifestd_tgs_pendantmdc_estimate_polars` and
    `alifestd_tgs_allmdc_estimate_polars`. Acronym tgs stands for trait
    gain/selection.

    Returns per-trait fits to bootstrap resamples of `num_items` exposure
    episodes, with columns `replicate` and `trait_index`, from `do_fit` (e.g., a
    `functools.partial` of `tgs_fit_traits`), called with each replicate's
    episode `weights`; see `run_weighted_bootstrap`.
    """
    fits = run_weighted_bootstrap(
        do_fit,
        num_items=num_items,
        num_replicates=num_bootstrap,
        seed=seed,
        parallel_backend=parallel_backend,
        progress_wrap=progress_wrap,
    )
    return pl.concat(
        replicate_fits.select(
            pl.lit(replicate, dtype=pl.Int64).alias("replicate"),
            pl.int_range(pl.len(), dtype=pl.Int64).alias("trait_index"),
            pl.all(),
        )
        for replicate, replicate_fits in enumerate(fits)
    )
