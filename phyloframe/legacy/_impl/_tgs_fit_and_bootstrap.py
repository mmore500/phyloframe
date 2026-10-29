import logging
import typing

import numpy as np
import pandas as pd

from ._tgs_bootstrap import tgs_bootstrap
from ._tgs_count_traits import tgs_count_traits
from ._tgs_fit_traits import tgs_fit_traits
from ._tgs_partition_leaves import tgs_partition_leaves
from ._tgs_select_calibration_neighborhoods import (
    tgs_select_calibration_neighborhoods,
)


def tgs_fit_and_bootstrap(
    *,
    res: pd.DataFrame,
    pendant_lengths: np.ndarray,
    origin_times: np.ndarray,
    trait_leaves: typing.Sequence[np.ndarray],
    trait_carriers: typing.Sequence[np.ndarray],
    trait_names: typing.Sequence[str],
    neutral_traits: typing.Sequence[int],
    calibrate: bool,
    calibration_neighborhood_fractions: typing.Sequence[float],
    num_calibration_partitions: int,
    num_bootstrap: int,
    loss_rate: float,
    seed: typing.Optional[int],
    parallel_backend: typing.Optional[str],
    progress_wrap: typing.Callable,
) -> typing.Optional[pd.DataFrame]:
    """Implementation detail for
    `alifestd_estimate_trait_gain_selection_pendant_asexual`.

    Here, tgs stands for trait gain/selection.

    Fits each trait to informative leaves, given their pendant lengths and
    origin times, filling point estimates into `res` in place, and returns
    bootstrap replicate fits (None without bootstrap replicates).
    """
    leaf_partitions, num_partitions, partition_coords = tgs_partition_leaves(
        pendant_lengths=pendant_lengths,
        origin_times=origin_times,
        max_partitions=num_calibration_partitions,
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
        loss_rate=loss_rate,
    )
    if calibrate:
        fit_kwargs[
            "calibration_neighborhoods"
        ] = tgs_select_calibration_neighborhoods(
            partition_coords=partition_coords,
            neighborhood_fractions=calibration_neighborhood_fractions,
            counts=tgs_count_traits(
                leaf_partitions=leaf_partitions,
                num_partitions=num_partitions,
                pendant_lengths=pendant_lengths,
                leaf_weights=np.ones(num_leaves),
                trait_leaves=trait_leaves,
                trait_carriers=trait_carriers,
            ),
            neutral_traits=neutral_traits,
            trait_names=trait_names,
        )

    logging.info(
        "- alifestd_estimate_trait_gain_selection_pendant_asexual: "
        f"fitting {len(trait_leaves)} traits to {num_leaves} leaves...",
    )
    point, calibration_factors = tgs_fit_traits(
        leaf_weights=np.ones(num_leaves), **fit_kwargs
    )
    res[list(point.columns)] = point
    for name, factors in zip(trait_names, calibration_factors):
        if factors is not None and len(factors):
            logging.info(
                "- alifestd_estimate_trait_gain_selection_pendant_asexual: "
                f"calibrated trait {name}, rescaling pendant lengths by "
                f"{factors.min():.3g} to {factors.max():.3g} (median "
                f"{np.median(factors):.3g})",
            )
    if not num_bootstrap:
        return None

    logging.info(
        "- alifestd_estimate_trait_gain_selection_pendant_asexual: "
        f"fitting {num_bootstrap} bootstrap replicates...",
    )
    return tgs_bootstrap(
        fit_kwargs=fit_kwargs,
        num_bootstrap=num_bootstrap,
        seed=seed,
        parallel_backend=parallel_backend,
        progress_wrap=progress_wrap,
    )
