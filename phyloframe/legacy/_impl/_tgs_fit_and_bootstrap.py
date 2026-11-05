import functools
import logging
import typing

import numpy as np
import pandas as pd

from ._tgs_bootstrap import tgs_bootstrap
from ._tgs_calc_rank_partitions import tgs_calc_rank_partitions
from ._tgs_count_traits import tgs_count_traits
from ._tgs_fit_traits import tgs_fit_traits
from ._tgs_select_pendant_calibration import tgs_select_pendant_calibration


def tgs_fit_and_bootstrap(
    *,
    pendant_lengths: np.ndarray,
    origin_times: np.ndarray,
    trait_leaves: typing.Sequence[np.ndarray],
    trait_carriers: typing.Sequence[np.ndarray],
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
    seed: typing.Optional[int],
    parallel_backend: typing.Optional[str],
    progress_wrap: typing.Callable,
) -> typing.Tuple[pd.DataFrame, typing.Optional[pd.DataFrame]]:
    """Implementation detail for `alifestd_tgs_pendantmdc_estimate_asexual`.
    Acronym tgs stands for trait gain/selection.

    Fits each trait to informative leaves, given their pendant lengths and
    origin times, returning point fits and bootstrap replicate fits (None
    without bootstrap replicates).
    """
    (
        leaf_partitions,
        num_partitions,
        partition_coords,
    ) = tgs_calc_rank_partitions(
        coordinates=[pendant_lengths, origin_times],
        max_partitions=calibrate_num_partitions,
    )
    num_leaves = len(leaf_partitions)
    fit_kwargs = dict(
        leaf_partitions=leaf_partitions,
        num_partitions=num_partitions,
        pendant_lengths=pendant_lengths,
        trait_leaves=trait_leaves,
        trait_carriers=trait_carriers,
        neutral_traits=neutral_traits,
        calibration_neighborhoods=None,
        max_rate_ratio=calibrate_max_rate_ratio,
        loss_rate=loss_rate,
        selection_starts=selection_starts,
        growth_exponent_bound=growth_exponent_bound,
    )
    if calibrate:
        fit_kwargs[
            "calibration_neighborhoods"
        ] = tgs_select_pendant_calibration(
            counts=tgs_count_traits(
                leaf_partitions=leaf_partitions,
                num_partitions=num_partitions,
                pendant_lengths=pendant_lengths,
                leaf_weights=np.ones(num_leaves),
                trait_leaves=trait_leaves,
                trait_carriers=trait_carriers,
            ),
            partition_coords=partition_coords,
            neighborhood_fractions=calibrate_neighborhood_fractions,
            neutral_traits=neutral_traits,
            trait_names=trait_names,
            max_rate_ratio=calibrate_max_rate_ratio,
        )

    logging.info(
        "- alifestd_tgs_pendantmdc_estimate_asexual: "
        f"fitting {len(trait_leaves)} traits to {num_leaves} leaves...",
    )
    point_df = tgs_fit_traits(weights=np.ones(num_leaves), **fit_kwargs)
    if not num_bootstrap:
        return point_df, None

    logging.info(
        "- alifestd_tgs_pendantmdc_estimate_asexual: "
        f"fitting {num_bootstrap} bootstrap replicates...",
    )
    return point_df, tgs_bootstrap(
        fit=functools.partial(tgs_fit_traits, **fit_kwargs),
        num_items=num_leaves,
        num_bootstrap=num_bootstrap,
        seed=seed,
        parallel_backend=parallel_backend,
        progress_wrap=progress_wrap,
    )
