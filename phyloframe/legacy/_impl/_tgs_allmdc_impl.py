import functools
import logging
import typing

import numpy as np
import pandas as pd

from ._tgs_bootstrap import tgs_bootstrap
from ._tgs_build_estimates_df import tgs_build_estimates_df
from ._tgs_fit_allmdc_traits import tgs_fit_allmdc_traits
from ._tgs_init_allmdc_results import tgs_init_allmdc_results
from ._tgs_partition_allmdc_exposures import tgs_partition_allmdc_exposures
from ._tgs_validate_allmdc_args import tgs_validate_allmdc_args


def tgs_allmdc_impl(
    exposure_df: pd.DataFrame,
    *,
    trait_names: typing.Sequence[str],
    neutral_traits: typing.Sequence[int],
    calibrate: bool,
    calibrate_neighborhood_fractions: typing.Sequence[float],
    calibrate_num_partitions: int,
    calibrate_size_partition_ratio: float,
    calibrate_size_partition_min_neutral_gains: int,
    num_survival_time_bins: int,
    num_bootstrap: int,
    loss_rate: float,
    selection_starts: typing.Sequence[float],
    growth_exponent_bound: float,
    seed: typing.Optional[int],
    parallel_backend: typing.Optional[str],
    progress_wrap: typing.Callable,
) -> pd.DataFrame:
    """Implementation detail for
    `alifestd_tgs_allmdc_estimate_asexual` and
    `alifestd_tgs_allmdc_estimate_polars`.

    Here, tgs stands for trait gain/selection.

    Takes `exposure_df` from `tgs_calc_exposure_df` and arguments as documented in
    `alifestd_tgs_allmdc_estimate_asexual`, with `neutral_traits` as
    indices and `trait_names` as labels (see `tgs_validate_args`).
    """
    tgs_validate_allmdc_args(
        num_survival_time_bins=num_survival_time_bins,
        calibrate_size_partition_ratio=calibrate_size_partition_ratio,
        calibrate_size_partition_min_neutral_gains=(
            calibrate_size_partition_min_neutral_gains
        ),
    )
    estimates_df = tgs_init_allmdc_results(
        trait_names=trait_names, exposure_df=exposure_df
    )
    num_branches = len(exposure_df)
    if num_branches:
        fit_kwargs = dict(
            exposure_df=exposure_df,
            partitioning=tgs_partition_allmdc_exposures(
                exposure_df=exposure_df,
                neutral_traits=neutral_traits,
                trait_names=trait_names,
                calibrate=calibrate,
                calibrate_neighborhood_fractions=(
                    calibrate_neighborhood_fractions
                ),
                calibrate_num_partitions=calibrate_num_partitions,
                calibrate_size_partition_ratio=calibrate_size_partition_ratio,
                calibrate_size_partition_min_neutral_gains=(
                    calibrate_size_partition_min_neutral_gains
                ),
                num_survival_time_bins=num_survival_time_bins,
            ),
            neutral_traits=neutral_traits,
            calibrate=calibrate,
            loss_rate=loss_rate,
            selection_starts=selection_starts,
            growth_exponent_bound=growth_exponent_bound,
        )
        logging.info(
            "- alifestd_tgs_allmdc_estimate_asexual: fitting "
            f"{len(trait_names)} traits to {num_branches} branches...",
        )
        point_df = tgs_fit_allmdc_traits(
            weights=np.ones(num_branches), **fit_kwargs
        )
        estimates_df[list(point_df.columns)] = point_df
        for __, row in estimates_df.iterrows():
            logging.info(
                f"- alifestd_tgs_allmdc_estimate_asexual: trait {row['trait']} "
                f"fit gain rate {row['gain_rate']:.4g}, selection "
                f"{row['selection']:.4g}, gain ratio {row['gain_ratio']:.4g}, "
                f"selection lrt {row['selection_lrt_naive']:.4g}",
            )
    replicate_fits_df = None
    if num_branches and num_bootstrap:
        logging.info(
            "- alifestd_tgs_allmdc_estimate_asexual: "
            f"fitting {num_bootstrap} bootstrap replicates...",
        )
        replicate_fits_df = tgs_bootstrap(
            fit=functools.partial(tgs_fit_allmdc_traits, **fit_kwargs),
            num_items=num_branches,
            num_bootstrap=num_bootstrap,
            seed=seed,
            parallel_backend=parallel_backend,
            progress_wrap=progress_wrap,
        )
    return tgs_build_estimates_df(
        estimates_df,
        replicate_fits_df=replicate_fits_df,
        num_bootstrap=num_bootstrap,
    )
