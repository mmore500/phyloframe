import logging
import typing

import numpy as np
import polars as pl

from ._tgs_build_estimates_df import tgs_build_estimates_df
from ._tgs_fit_traits import tgs_fit_traits
from ._tgs_prepare_trait_fits import tgs_prepare_trait_fits


def tgs_estimate_tgs_from_exposures(
    exposure_df: pl.DataFrame,
    *,
    counts_df: pl.DataFrame,
    trait_names: typing.Sequence[str],
    neutral_traits: typing.Sequence[int],
    calibrate_neighborhood_fractions: typing.Sequence[float],
    calibrate_max_rate_ratio: float,
    length_rtol: float,
    loss_rate: float,
    selection_coef_starts: typing.Sequence[float],
    growth_exponent_bound: float,
    progress_wrap: typing.Callable,
) -> pl.DataFrame:
    """Implementation detail for `alifestd_tgs_pendantmdc_estimate_polars` and
    `alifestd_tgs_allmdc_estimate_polars`. Acronym tgs stands for trait
    gain/selection.

    Estimates each trait's gain rate and selection coefficient from the
    exposure episodes of `exposure_df` (see `tgs_build_exposure_df`),
    setting up calibration (see `tgs_prepare_trait_fits`) and fitting each
    trait (see `tgs_fit_traits`), with progress over traits wrapped by
    `progress_wrap`. Returns per-trait estimates following `counts_df`'s
    per-trait counts; see `tgs_build_estimates_df`. Takes validated
    arguments, with `neutral_traits` as indices and `trait_names` as labels
    (see `tgs_validate_args`).
    """
    fit_kwargs, trait_bin_counts_df = tgs_prepare_trait_fits(
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
    logging.info(
        f"- tgs: fitting {len(trait_names)} traits to {len(exposure_df)} "
        "exposure episodes...",
    )
    return tgs_build_estimates_df(
        counts_df,
        point_estimate_df=tgs_fit_traits(
            weights=np.ones(len(exposure_df)),
            progress_wrap=progress_wrap,
            trait_bin_counts_df=trait_bin_counts_df,
            **fit_kwargs,
        ),
    )
