import functools
import logging
import typing

import numpy as np
import pandas as pd

from ._tgs_allmdc_count_episodes import tgs_allmdc_count_episodes
from ._tgs_allmdc_fit_traits import tgs_allmdc_fit_traits
from ._tgs_allmdc_partition_episodes import tgs_allmdc_partition_episodes
from ._tgs_allmdc_validate_args import tgs_allmdc_validate_args
from ._tgs_bootstrap import tgs_bootstrap
from ._tgs_build_estimates_df import tgs_build_estimates_df


def tgs_allmdc_impl(
    exposure_df: pd.DataFrame,
    *,
    trait_names: typing.Sequence[str],
    neutral_traits: typing.Sequence[int],
    calibrate: bool,
    calibrate_neighborhood_fractions: typing.Sequence[float],
    calibrate_num_partitions: int,
    calibrate_max_rate_ratio: float,
    calibrate_size_partition_ratio: float,
    calibrate_size_partition_min_neutral_gains: int,
    num_bootstrap: int,
    loss_rate: float,
    selection_starts: typing.Sequence[float],
    growth_exponent_bound: float,
    seed: typing.Optional[int],
    parallel_backend: typing.Optional[str],
    progress_wrap: typing.Callable,
) -> pd.DataFrame:
    """Implementation detail for `alifestd_tgs_allmdc_estimate_polars`. Acronym
    tgs stands for trait gain/selection.

    Takes `exposure_df` from `tgs_calc_exposure_df` and arguments as documented
    in `alifestd_tgs_allmdc_estimate_polars`, with `neutral_traits` as indices
    and `trait_names` as labels (see `tgs_validate_args`).
    """
    tgs_allmdc_validate_args(
        calibrate_size_partition_ratio=calibrate_size_partition_ratio,
        calibrate_size_partition_min_neutral_gains=(
            calibrate_size_partition_min_neutral_gains
        ),
    )
    num_episodes = len(exposure_df)
    if not num_episodes:
        raise ValueError(
            "phylogeny_df has no exposure episodes, i.e., branches to fit",
        )
    fit_kwargs = dict(
        exposure_df=exposure_df,
        partitioning=tgs_allmdc_partition_episodes(
            exposure_df=exposure_df,
            neutral_traits=neutral_traits,
            trait_names=trait_names,
            calibrate=calibrate,
            calibrate_neighborhood_fractions=calibrate_neighborhood_fractions,
            calibrate_num_partitions=calibrate_num_partitions,
            calibrate_max_rate_ratio=calibrate_max_rate_ratio,
            calibrate_size_partition_ratio=calibrate_size_partition_ratio,
            calibrate_size_partition_min_neutral_gains=(
                calibrate_size_partition_min_neutral_gains
            ),
        ),
        neutral_traits=neutral_traits,
        calibrate=calibrate,
        calibrate_max_rate_ratio=calibrate_max_rate_ratio,
        loss_rate=loss_rate,
        selection_starts=selection_starts,
        growth_exponent_bound=growth_exponent_bound,
    )
    logging.info(
        "- alifestd_tgs_allmdc_estimate_polars: fitting "
        f"{len(trait_names)} traits to {num_episodes} exposure episodes...",
    )
    point_estimate_df = tgs_allmdc_fit_traits(
        weights=np.ones(num_episodes), **fit_kwargs
    )
    for name, (__, row) in zip(trait_names, point_estimate_df.iterrows()):
        logging.info(
            f"- alifestd_tgs_allmdc_estimate_polars: trait {name} "
            f"fit gain rate {row['gain_rate']:.4g}, selection "
            f"{row['selection']:.4g}, gain ratio {row['gain_ratio']:.4g}, "
            f"selection lrt {row['selection_lrt_naive']:.4g}",
        )
    bootstrap_fits_df = None
    if num_bootstrap:
        logging.info(
            "- alifestd_tgs_allmdc_estimate_polars: "
            f"fitting {num_bootstrap} bootstrap replicates...",
        )
        bootstrap_fits_df = tgs_bootstrap(
            fit=functools.partial(tgs_allmdc_fit_traits, **fit_kwargs),
            num_items=num_episodes,
            num_bootstrap=num_bootstrap,
            seed=seed,
            parallel_backend=parallel_backend,
            progress_wrap=progress_wrap,
        )
    return tgs_build_estimates_df(
        tgs_allmdc_count_episodes(
            trait_names=trait_names, exposure_df=exposure_df
        ),
        point_estimate_df=point_estimate_df,
        bootstrap_fits_df=bootstrap_fits_df,
        num_bootstrap=num_bootstrap,
    )
