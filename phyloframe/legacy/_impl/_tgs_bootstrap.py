import functools
import gc
import logging
import typing

import polars as pl

from ..._auxlib._identity import identity
from ..._auxlib._log_memory_usage import log_memory_usage
from ._tgs_bootstrap_trait_fits import tgs_bootstrap_trait_fits
from ._tgs_summarize_bootstrap import tgs_summarize_bootstrap


def _tgs_fit_replicate(
    estimate_impl: typing.Callable[..., typing.Dict[str, typing.Any]],
    *,
    weights: typing.Any,
    **estimate_kwargs: typing.Any,
) -> pl.DataFrame:
    """Implementation detail for `tgs_bootstrap`. Acronym tgs stands for trait
    gain/selection.

    Returns one bootstrap replicate's per-trait fits, from `estimate_impl`
    with the replicate's episode `weights`. Module level, so picklable for
    process-based backends.
    """
    return estimate_impl(None, weights=weights, **estimate_kwargs)["fits_df"]


def tgs_bootstrap(
    estimate_impl: typing.Callable[..., typing.Dict[str, typing.Any]],
    phylogeny_df: pl.DataFrame,
    *,
    estimate_kwargs: typing.Dict[str, typing.Any],
    num_bootstrap: int,
    seed: typing.Optional[int],
    parallel_backend: typing.Optional[str],
    progress_wrap: typing.Callable,
) -> pl.DataFrame:
    """Implementation detail for
    `alifestd_tgs_pendantmdc_estimate_bootstrap_polars` and
    `alifestd_tgs_allmdc_estimate_bootstrap_polars`. Acronym tgs stands for
    trait gain/selection.

    Point estimates `phylogeny_df` with `estimate_impl` (e.g., the private
    implementation of `alifestd_tgs_pendantmdc_estimate_polars`), called
    with `estimate_kwargs`, then refits it to `num_bootstrap` resamples of
    the point estimate's exposure episodes, reusing its exposure table and
    holding its calibration neighborhoods fixed; see
    `tgs_bootstrap_trait_fits`. `progress_wrap` wraps the point estimate's
    traits, then the replicates.

    `estimate_impl` takes `phylogeny_df` positionally; `exposure_df`,
    `calibration_neighborhoods`, and `weights`, all None for a point
    estimate, else all given; `progress_wrap`; and `estimate_kwargs`. It
    returns a dict of `estimates_df`, `fits_df`, `exposure_df`, and
    `calibration_neighborhoods`.

    Returns the point estimate's `estimates_df` with bootstrap standard
    errors, z-scores, and p-values appended; see `tgs_summarize_bootstrap`.
    """
    if not num_bootstrap > 0:
        raise ValueError(
            f"num_bootstrap must be positive, got {num_bootstrap}",
        )
    point = estimate_impl(
        phylogeny_df,
        exposure_df=None,
        calibration_neighborhoods=None,
        weights=None,
        progress_wrap=progress_wrap,
        **estimate_kwargs,
    )
    logging.info(f"- tgs: fitting {num_bootstrap} bootstrap replicates...")
    bootstrap_fits_df = tgs_bootstrap_trait_fits(
        do_fit=functools.partial(
            _tgs_fit_replicate,
            estimate_impl,
            exposure_df=point["exposure_df"],  # reused, not rebuilt
            calibration_neighborhoods=point["calibration_neighborhoods"],
            progress_wrap=identity,  # progress is over replicates
            **estimate_kwargs,
        ),
        num_items=len(point["exposure_df"]),
        num_bootstrap=num_bootstrap,
        seed=seed,
        parallel_backend=parallel_backend,
        progress_wrap=progress_wrap,
    )
    gc.collect()
    log_memory_usage(logging.info)
    estimates_df = point["estimates_df"]
    return pl.concat(
        [
            estimates_df,
            tgs_summarize_bootstrap(
                point_estimate_df=estimates_df,
                bootstrap_fits_df=bootstrap_fits_df,
            ),
        ],
        how="horizontal",
    )
