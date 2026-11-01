import logging
import typing

import numpy as np
import pandas as pd
import polars as pl

from ._tgs_bootstrap import tgs_bootstrap
from ._tgs_find_allmdc_branches import tgs_find_allmdc_branches
from ._tgs_fit_allmdc_bootstrap_replicate import (
    tgs_fit_allmdc_bootstrap_replicate,
)
from ._tgs_fit_allmdc_traits import tgs_fit_allmdc_traits
from ._tgs_init_allmdc_results import tgs_init_allmdc_results
from ._tgs_partition_allmdc_branches import tgs_partition_allmdc_branches
from ._tgs_summarize_bootstrap import tgs_summarize_bootstrap


def tgs_allmdc_impl(
    phylogeny_df: typing.Union[pd.DataFrame, pl.DataFrame],
    *,
    trait_names: typing.Sequence[str],
    neutral_traits: typing.Sequence[int],
    calibrate: bool,
    calibration_neighborhood_fractions: typing.Sequence[float],
    num_calibration_partitions: int,
    clade_size_calibration_blocks: typing.Sequence[int],
    num_survival_time_bins: int,
    num_bootstrap: int,
    loss_rate: float,
    selection_starts: typing.Sequence[float],
    max_growth_exponent: float,
    seed: typing.Optional[int],
    parallel_backend: typing.Optional[str],
    progress_wrap: typing.Callable,
) -> pd.DataFrame:
    """Implementation detail for
    `alifestd_tgs_allmdc_estimate_asexual` and
    `alifestd_tgs_allmdc_estimate_polars`.

    Here, tgs stands for trait gain/selection.

    Takes a phylogeny prepared by `tgs_prepare_asexual` or
    `tgs_prepare_polars`, with contiguous, topologically sorted ids and no
    unifurcations, and validated arguments (see `tgs_validate_args`).
    """
    if num_survival_time_bins < 1:
        raise ValueError("num_survival_time_bins must be at least 1")
    blocks = np.asarray(clade_size_calibration_blocks)
    if blocks.size and (blocks[0] < 2 or (np.diff(blocks) <= 0).any()):
        raise ValueError(
            "clade_size_calibration_blocks must be strictly increasing clade "
            "sizes of at least 2",
        )
    branches = tgs_find_allmdc_branches(
        ancestor_ids=phylogeny_df["ancestor_id"].to_numpy(),
        origin_time=phylogeny_df["origin_time"].to_numpy(),
        branch_lengths=phylogeny_df["__alifestd_tgs_branch_length"].to_numpy(),
        trait_masks=[
            phylogeny_df[f"__alifestd_tgs_trait{i}"].to_numpy()
            for i in range(len(trait_names))
        ],
    )
    res = tgs_init_allmdc_results(trait_names=trait_names, branches=branches)
    num_branches = len(branches.branch_lengths)
    if num_branches:
        fit_kwargs = dict(
            branches=branches,
            partitioning=tgs_partition_allmdc_branches(
                branches=branches,
                neutral_traits=neutral_traits,
                trait_names=trait_names,
                calibrate=calibrate,
                calibration_neighborhood_fractions=(
                    calibration_neighborhood_fractions
                ),
                num_calibration_partitions=num_calibration_partitions,
                clade_size_calibration_blocks=clade_size_calibration_blocks,
                num_survival_time_bins=num_survival_time_bins,
            ),
            neutral_traits=neutral_traits,
            calibrate=calibrate,
            loss_rate=loss_rate,
            selection_starts=selection_starts,
            max_growth_exponent=max_growth_exponent,
        )
        logging.info(
            "- alifestd_tgs_allmdc_estimate_asexual: fitting "
            f"{len(trait_names)} traits to {num_branches} branches...",
        )
        point = tgs_fit_allmdc_traits(
            unit_weights=np.ones(num_branches), **fit_kwargs
        )
        res[list(point.columns)] = point
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
