import operator
import typing

import numpy as np
import pandas as pd

from .._alifestd_mark_leaves import _alifestd_mark_leaves_asexual_fast_path
from ._tgs_find_trait_leaves import tgs_find_trait_leaves
from ._tgs_fit_and_bootstrap import tgs_fit_and_bootstrap
from ._tgs_init_results import tgs_init_results
from ._tgs_summarize_bootstrap import tgs_summarize_bootstrap


def tgs_impl(
    *,
    ancestor_ids: np.ndarray,
    origin_time: np.ndarray,
    pendant_lengths: np.ndarray,
    trait_masks: typing.Sequence[np.ndarray],
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
    `alifestd_estimate_trait_gain_selection_pendant_asexual` and
    `alifestd_estimate_trait_gain_selection_pendant_polars`.

    Here, tgs stands for trait gain/selection.

    Takes per-node arrays for a phylogeny with contiguous, topologically
    sorted ids and no unifurcations: ancestor ids, origin times, pendant
    branch lengths (origin time deltas), and boolean trait masks.
    """
    neutral_traits = np.unique(
        np.arange(len(trait_masks))[list(map(operator.index, neutral_traits))]
    ).tolist()
    if calibrate and len(neutral_traits) < 2:
        raise ValueError("calibration requires at least two neutral traits")
    if not calibration_neighborhood_fractions or not all(
        0 < fraction <= 1 for fraction in calibration_neighborhood_fractions
    ):
        raise ValueError(
            "calibration_neighborhood_fractions must be nonempty, with each "
            "in (0, 1]",
        )
    if int(num_calibration_partitions).bit_count() != 1 or (
        num_calibration_partitions < 1
    ):
        raise ValueError(
            "num_calibration_partitions must be a power of two, got "
            f"{num_calibration_partitions}",
        )

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
