import typing

import pandas as pd

from ._tgs_build_estimates_df import tgs_build_estimates_df
from ._tgs_calc_trait_masks import tgs_calc_trait_masks
from ._tgs_find_trait_exposures import tgs_find_trait_exposures
from ._tgs_fit_and_bootstrap import tgs_fit_and_bootstrap


def tgs_impl(
    exposure_df: pd.DataFrame,
    *,
    trait_names: typing.Sequence[str],
    neutral_traits: typing.Sequence[int],
    calibrate: bool,
    calibrate_neighborhood_fractions: typing.Sequence[float],
    calibrate_num_partitions: int,
    calibrate_max_rate_ratio: float,
    num_bootstrap: int,
    loss_rate: float,
    selection_starts: typing.Sequence[float],
    growth_exponent_bound: float,
    min_leaf_origin_time: float,
    seed: typing.Optional[int],
    parallel_backend: typing.Optional[str],
    progress_wrap: typing.Callable,
) -> pd.DataFrame:
    """Implementation detail for `alifestd_tgs_pendantmdc_estimate_asexual` and
    `alifestd_tgs_pendantmdc_estimate_polars`. Acronym tgs stands for trait
    gain/selection.

    Takes `exposure_df` from `tgs_calc_exposure_df` and validated arguments (see
    `tgs_validate_args`), and fits leaf pendant branches originating at or
    after `min_leaf_origin_time`.
    """
    leaf_exposure_df = exposure_df[
        (exposure_df["clade_size"] == 1)
        & (exposure_df["origin_time"] >= min_leaf_origin_time)
    ]
    trait_leaves, trait_carriers = tgs_find_trait_exposures(leaf_exposure_df)
    starts_absent, gains = tgs_calc_trait_masks(leaf_exposure_df)
    counts_df = pd.DataFrame(
        {
            "trait": list(trait_names),
            "num_leaves": starts_absent.sum(axis=0),
            "num_trait_present": gains.sum(axis=0),
        },
    )
    point_df, replicate_fits_df = None, None
    if len(leaf_exposure_df):
        point_df, replicate_fits_df = tgs_fit_and_bootstrap(
            pendant_lengths=leaf_exposure_df["branch_length"].to_numpy(),
            origin_times=leaf_exposure_df["origin_time"].to_numpy(),
            trait_leaves=trait_leaves,
            trait_carriers=trait_carriers,
            trait_names=trait_names,
            neutral_traits=neutral_traits,
            calibrate=calibrate,
            calibrate_neighborhood_fractions=calibrate_neighborhood_fractions,
            calibrate_num_partitions=calibrate_num_partitions,
            calibrate_max_rate_ratio=calibrate_max_rate_ratio,
            num_bootstrap=num_bootstrap,
            loss_rate=loss_rate,
            selection_starts=selection_starts,
            growth_exponent_bound=growth_exponent_bound,
            seed=seed,
            parallel_backend=parallel_backend,
            progress_wrap=progress_wrap,
        )
    return tgs_build_estimates_df(
        counts_df,
        point_df=point_df,
        replicate_fits_df=replicate_fits_df,
        num_bootstrap=num_bootstrap,
    )
