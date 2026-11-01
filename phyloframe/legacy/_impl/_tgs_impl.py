import typing

import numpy as np
import pandas as pd
import polars as pl

from .._alifestd_mark_leaves import _alifestd_mark_leaves_asexual_fast_path
from ._tgs_find_trait_leaves import tgs_find_trait_leaves
from ._tgs_fit_and_bootstrap import tgs_fit_and_bootstrap
from ._tgs_init_results import tgs_init_results
from ._tgs_summarize_bootstrap import tgs_summarize_bootstrap


def tgs_impl(
    phylogeny_df: typing.Union[pd.DataFrame, pl.DataFrame],
    *,
    trait_names: typing.Sequence[str],
    neutral_traits: typing.Sequence[int],
    calibrate: bool,
    calibration_neighborhood_fractions: typing.Sequence[float],
    num_calibration_partitions: int,
    num_bootstrap: int,
    loss_rate: float,
    min_leaf_origin_time: float,
    seed: typing.Optional[int],
    parallel_backend: typing.Optional[str],
    progress_wrap: typing.Callable,
) -> pd.DataFrame:
    """Implementation detail for
    `alifestd_tgs_pendantmdc_estimate_asexual` and
    `alifestd_tgs_pendantmdc_estimate_polars`.

    Here, tgs stands for trait gain/selection.

    Takes a phylogeny prepared by `tgs_prepare_asexual` or
    `tgs_prepare_polars`, with contiguous, topologically sorted ids and no
    unifurcations, and validated arguments (see `tgs_validate_args`).
    """
    ancestor_ids = phylogeny_df["ancestor_id"].to_numpy()
    origin_time = phylogeny_df["origin_time"].to_numpy()
    pendant_lengths = phylogeny_df["__alifestd_tgs_branch_length"].to_numpy()
    trait_masks = [
        phylogeny_df[f"__alifestd_tgs_trait{i}"].to_numpy()
        for i in range(len(trait_names))
    ]
    ancestor_ids = np.asarray(ancestor_ids, dtype=np.int64)
    origin_time = np.asarray(origin_time, dtype=np.float64)
    pendant_lengths = np.asarray(pendant_lengths, dtype=np.float64)
    # leaves informative for fitting: those with a pendant branch to gain a
    # trait on, originating at or after `min_leaf_origin_time`
    is_informative = (
        _alifestd_mark_leaves_asexual_fast_path(ancestor_ids)
        & (pendant_lengths > 0)  # excludes root leaves and nan
        & (origin_time >= min_leaf_origin_time)
    )
    # for each trait, informative leaves whose pendant branch started without
    # the trait, and the pendant gains among them
    trait_leaves, trait_carriers = [], []
    for trait_mask in trait_masks:
        leaves, carriers = tgs_find_trait_leaves(
            ancestor_ids=ancestor_ids,
            trait_mask=trait_mask,
            is_informative=is_informative,
        )
        trait_leaves.append(leaves)
        trait_carriers.append(carriers)
    res = tgs_init_results(
        trait_names=trait_names,
        trait_leaves=trait_leaves,
        trait_carriers=trait_carriers,
    )
    if is_informative.any():
        replicate_fits = tgs_fit_and_bootstrap(
            res=res,
            pendant_lengths=pendant_lengths[is_informative],
            origin_times=origin_time[is_informative],
            trait_leaves=trait_leaves,
            trait_carriers=trait_carriers,
            trait_names=trait_names,
            neutral_traits=neutral_traits,
            calibrate=calibrate,
            calibration_neighborhood_fractions=calibration_neighborhood_fractions,
            num_calibration_partitions=num_calibration_partitions,
            num_bootstrap=num_bootstrap,
            loss_rate=loss_rate,
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

    if not num_bootstrap:
        return res
    return pd.concat(
        [
            res,
            tgs_summarize_bootstrap(point=res, replicate_fits=replicate_fits),
        ],
        axis=1,
    )
