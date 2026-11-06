import functools
import logging
import typing

import numpy as np
import pandas as pd
from scipy import sparse

from ..._auxlib._calc_group_means import calc_group_means
from ..._auxlib._calc_tiebroken_ranks import calc_tiebroken_ranks
from ._tgs_bootstrap_trait_fits import tgs_bootstrap_trait_fits
from ._tgs_build_estimates_df import tgs_build_estimates_df
from ._tgs_count_traits import tgs_count_traits
from ._tgs_fit_traits import tgs_fit_traits
from ._tgs_select_calibration_neighborhoods import (
    tgs_select_calibration_neighborhoods,
)


def tgs_impl(
    exposure_df: pd.DataFrame,
    *,
    counts_df: pd.DataFrame,
    trait_names: typing.Sequence[str],
    neutral_traits: typing.Sequence[int],
    calibrate_neighborhood_fractions: typing.Sequence[float],
    calibrate_max_rate_ratio: float,
    length_rtol: float,
    num_bootstrap: int,
    loss_rate: float,
    selection_coef_starts: typing.Sequence[float],
    growth_exponent_bound: float,
    seed: typing.Optional[int],
    parallel_backend: typing.Optional[str],
    progress_wrap: typing.Callable,
) -> pd.DataFrame:
    """Implementation detail for `alifestd_tgs_pendantmdc_estimate_polars` and
    `alifestd_tgs_allmdc_estimate_polars`. Acronym tgs stands for trait
    gain/selection.

    Fits each trait to the exposure episodes of `exposure_df` (see
    `tgs_calc_exposure_df`), calibrating within its `calibration_partition`
    column, if any, and returns per-trait estimates following `counts_df`'s
    per-trait counts (see `tgs_build_estimates_df`). Takes validated
    arguments, with `neutral_traits` as indices and `trait_names` as labels
    (see `tgs_validate_args`).
    """
    num_episodes = len(exposure_df)
    if not num_episodes:
        raise ValueError("phylogeny_df has no exposure episodes to fit")

    # number calibration partitions consecutively; without them, all are one
    __, partitions = np.unique(
        exposure_df.get(
            "calibration_partition", np.zeros(num_episodes, dtype=np.int64)
        ),
        return_inverse=True,
    )
    num_partitions = partitions.max() + 1
    if num_partitions > 1:
        # neighborhoods are nearest partitions by their episodes' mean ranks;
        # constant coordinates (e.g., leaves' clade sizes) carry no distance
        coordinates = [
            exposure_df[column].to_numpy()
            for column in ("branch_length", "origin_time", "clade_size")
        ]
        ranks = calc_tiebroken_ranks(
            [x for x in coordinates if np.ptp(x) > 0] or coordinates[:1]
        )
        calibration_neighborhoods = tgs_select_calibration_neighborhoods(
            partition_coords=calc_group_means(
                partitions, ranks / num_episodes, num_groups=num_partitions
            ),
            neighborhood_fractions=calibrate_neighborhood_fractions,
            counts=tgs_count_traits(
                exposure_df=exposure_df,
                partitions=partitions,
                num_partitions=num_partitions,
                weights=np.ones(num_episodes),
            ),
            neutral_traits=neutral_traits,
            trait_names=trait_names,
            max_rate_ratio=calibrate_max_rate_ratio,
        )
    else:  # one neighborhood of all episodes, i.e., no calibration
        calibration_neighborhoods = sparse.identity(1, format="csr")

    distinct_survival_durations, survival_duration_indices = np.unique(
        exposure_df["survival_duration"].to_numpy(), return_inverse=True
    )
    fit_kwargs = dict(
        exposure_df=exposure_df,
        partitions=partitions,
        num_partitions=num_partitions,
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
    logging.info(
        f"- tgs: fitting {len(trait_names)} traits to {num_episodes} "
        "exposure episodes...",
    )
    point_estimate_df = tgs_fit_traits(
        weights=np.ones(num_episodes), **fit_kwargs
    )
    bootstrap_fits_df = None
    if num_bootstrap:
        logging.info(f"- tgs: fitting {num_bootstrap} bootstrap replicates...")
        bootstrap_fits_df = tgs_bootstrap_trait_fits(
            do_fit=functools.partial(tgs_fit_traits, **fit_kwargs),
            num_items=num_episodes,
            num_bootstrap=num_bootstrap,
            seed=seed,
            parallel_backend=parallel_backend,
            progress_wrap=progress_wrap,
        )
    return tgs_build_estimates_df(
        counts_df,
        point_estimate_df=point_estimate_df,
        bootstrap_fits_df=bootstrap_fits_df,
        num_bootstrap=num_bootstrap,
    )
