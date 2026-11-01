import logging
import typing

import numpy as np
import pandas as pd

from ._tgs_bootstrap import tgs_bootstrap
from ._tgs_fit_allmdc_bootstrap_replicate import (
    tgs_fit_allmdc_bootstrap_replicate,
)
from ._tgs_fit_allmdc_traits import tgs_fit_allmdc_traits
from ._tgs_init_allmdc_results import tgs_init_allmdc_results
from ._tgs_partition_allmdc_branches import tgs_partition_allmdc_branches
from ._tgs_summarize_bootstrap import tgs_summarize_bootstrap


def tgs_allmdc_impl(
    branches: pd.DataFrame,
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

    Takes branches from `tgs_calc_branches` and arguments as documented in
    `alifestd_tgs_allmdc_estimate_asexual`, with `neutral_traits` as
    indices and `trait_names` as labels (see `tgs_validate_args`).
    """
    if num_survival_time_bins < 1:
        raise ValueError("num_survival_time_bins must be at least 1")
    if not calibrate_size_partition_ratio > 1:
        raise ValueError("calibrate_size_partition_ratio must exceed 1")
    if calibrate_size_partition_min_neutral_gains < 1:
        raise ValueError(
            "calibrate_size_partition_min_neutral_gains must be at least 1"
        )
    res = tgs_init_allmdc_results(trait_names=trait_names, branches=branches)
    num_branches = len(branches)
    if num_branches:
        fit_kwargs = dict(
            branches=branches,
            partitioning=tgs_partition_allmdc_branches(
                branches=branches,
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
        point = tgs_fit_allmdc_traits(
            unit_weights=np.ones(num_branches), **fit_kwargs
        )
        res[list(point.columns)] = point
        for __, row in res.iterrows():
            logging.info(
                f"- alifestd_tgs_allmdc_estimate_asexual: trait {row['trait']} "
                f"fit gain rate {row['gain_rate']:.4g}, selection "
                f"{row['selection']:.4g}, gain ratio {row['gain_ratio']:.4g}, "
                f"selection lrt {row['selection_lrt_naive']:.4g}",
            )
    if not num_bootstrap:
        return res

    if num_branches:
        logging.info(
            "- alifestd_tgs_allmdc_estimate_asexual: "
            f"fitting {num_bootstrap} bootstrap replicates...",
        )
        replicate_fits = tgs_bootstrap(
            fit_replicate=tgs_fit_allmdc_bootstrap_replicate,
            fit_kwargs=fit_kwargs,
            num_bootstrap=num_bootstrap,
            seed=seed,
            parallel_backend=parallel_backend,
            progress_wrap=progress_wrap,
        )
    else:  # nothing to fit, so every replicate matches the empty fit
        replicate_fits = pd.concat(
            [res] * num_bootstrap,
            keys=range(num_bootstrap),
            names=["replicate", "trait_index"],
        )
    return pd.concat(
        [
            res,
            tgs_summarize_bootstrap(point=res, replicate_fits=replicate_fits),
        ],
        axis=1,
    )
