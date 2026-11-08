import functools
import logging
import typing

import polars as pl

from ._tgs_bootstrap_trait_fits import tgs_bootstrap_trait_fits
from ._tgs_fit_traits import tgs_fit_traits
from ._tgs_prepare_trait_fits import tgs_prepare_trait_fits


def tgs_bootstrap_tgs_from_exposures(
    exposure_df: pl.DataFrame,
    *,
    trait_names: typing.Sequence[str],
    neutral_traits: typing.Sequence[int],
    calibrate_neighborhood_fractions: typing.Sequence[float],
    calibrate_max_rate_ratio: float,
    length_rtol: float,
    loss_rate: float,
    selection_coef_starts: typing.Sequence[float],
    growth_exponent_bound: float,
    num_bootstrap: int,
    seed: typing.Optional[int],
    parallel_backend: typing.Optional[str],
    progress_wrap: typing.Callable,
) -> pl.DataFrame:
    """Implementation detail for `alifestd_tgs_pendantmdc_bootstrap_polars`
    and `alifestd_tgs_allmdc_bootstrap_polars`. Acronym tgs stands for trait
    gain/selection.

    Fits each trait to `num_bootstrap` bootstrap resamples of the exposure
    episodes of `exposure_df` (see `tgs_bootstrap_trait_fits`), with
    calibration set up once, from unweighted episodes, as for the point
    estimate (see `tgs_prepare_trait_fits`), so held fixed across
    replicates; see Output Schema.

    Output Schema
    -------------
    'replicate' : integer
        Bootstrap replicate index.
    'trait' : string
        Trait name, from `trait_names`.
    'gain_rate', 'selection_coef', 'selection_coef_lrt_naive', 'gain_ratio', 'gain_ratio_lrt_naive' : float
        The replicate's fits; see `tgs_fit_traits`.
    """
    fit_kwargs, __ = tgs_prepare_trait_fits(
        exposure_df,
        trait_names=trait_names,
        neutral_traits=neutral_traits,
        calibrate_neighborhood_fractions=calibrate_neighborhood_fractions,
        calibrate_max_rate_ratio=calibrate_max_rate_ratio,
        length_rtol=length_rtol,
        loss_rate=loss_rate,
        selection_coef_starts=selection_coef_starts,
        growth_exponent_bound=growth_exponent_bound,
    )
    logging.info(f"- tgs: fitting {num_bootstrap} bootstrap replicates...")
    return (
        tgs_bootstrap_trait_fits(
            do_fit=functools.partial(tgs_fit_traits, **fit_kwargs),
            num_items=len(exposure_df),
            num_bootstrap=num_bootstrap,
            seed=seed,
            parallel_backend=parallel_backend,
            progress_wrap=progress_wrap,
        )
        .with_columns(
            trait=pl.lit(pl.Series(list(trait_names), dtype=pl.String)).gather(
                pl.col("trait_index")
            ),
        )
        .select(
            "replicate",
            "trait",
            pl.exclude("replicate", "trait", "trait_index"),
        )
    )
