import typing

import numpy as np
import polars as pl

from ._tgs_assign_calibration_bins import tgs_assign_calibration_bins
from ._tgs_select_partitioned_calibration_neighborhoods import (
    tgs_select_partitioned_calibration_neighborhoods,
)


def tgs_prepare_trait_fits(
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
) -> typing.Tuple[typing.Dict[str, typing.Any], pl.DataFrame]:
    """Implementation detail for `alifestd_tgs_pendantmdc_estimate_polars`,
    `alifestd_tgs_allmdc_estimate_polars`, and their bootstrap counterparts.
    Acronym tgs stands for trait gain/selection.

    Sets up fitting traits to the exposure episodes of `exposure_df` (see
    `tgs_build_exposure_df`): assigns episodes to calibration bins (see
    `tgs_assign_calibration_bins`) and selects calibration neighborhoods
    within partitions (see `tgs_select_partitioned_calibration_neighborhoods`),
    once, from unweighted episodes. Returns `tgs_fit_traits` arguments but
    `weights`, shared by the point estimate and every bootstrap replicate,
    and the unweighted per-(trait, bin) counts, for reuse by the point
    estimate.
    """
    bins_df = tgs_assign_calibration_bins(exposure_df)
    bins = bins_df["bin"].to_numpy()
    num_bins = int(bins.max()) + 1
    (
        calibration_neighborhoods,
        trait_bin_counts_df,
    ) = tgs_select_partitioned_calibration_neighborhoods(
        exposure_df,
        bins_df=bins_df,
        num_bins=num_bins,
        neighborhood_fractions=calibrate_neighborhood_fractions,
        neutral_traits=neutral_traits,
        trait_names=trait_names,
        max_rate_ratio=calibrate_max_rate_ratio,
    )
    distinct_survival_durations, survival_duration_indices = np.unique(
        exposure_df["survival_duration"].to_numpy(), return_inverse=True
    )
    fit_kwargs = dict(
        exposure_df=exposure_df,
        bins=bins,
        num_bins=num_bins,
        calibration_neighborhoods=calibration_neighborhoods,
        distinct_survival_durations=distinct_survival_durations,
        survival_duration_indices=survival_duration_indices,
        neutral_traits=neutral_traits,
        calibrate_max_rate_ratio=calibrate_max_rate_ratio,
        length_rtol=length_rtol,
        loss_rate=loss_rate,
        selection_coef_starts=selection_coef_starts,
        growth_exponent_bound=growth_exponent_bound,
    )
    return fit_kwargs, trait_bin_counts_df
