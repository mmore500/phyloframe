import functools
import logging
import typing

import numpy as np
import pandas as pd

from ._tgs_bootstrap import tgs_bootstrap
from ._tgs_calc_calibrated_pendant_lengths import (
    tgs_calc_calibrated_pendant_lengths,
)
from ._tgs_count_traits import tgs_count_traits
from ._tgs_fit_traits import tgs_fit_traits
from ._tgs_partition_branches import tgs_partition_branches
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
    calibrate_neighborhood_fractions: typing.Sequence[float],
    calibrate_num_partitions: int,
    num_bootstrap: int,
    loss_rate: float,
    seed: typing.Optional[int],
    parallel_backend: typing.Optional[str],
    progress_wrap: typing.Callable,
) -> typing.Optional[pd.DataFrame]:
    """Implementation detail for
    `alifestd_tgs_pendantmdc_estimate_asexual`.

    Here, tgs stands for trait gain/selection.

    Fits each trait to informative leaves, given their pendant lengths and
    origin times, filling point estimates into `res` in place, and returns
    bootstrap replicate fits (None without bootstrap replicates).
    """
    leaf_partitions, num_partitions, partition_coords = tgs_partition_branches(
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
        loss_rate=loss_rate,
    )
    if calibrate:
        counts = tgs_count_traits(
            leaf_partitions=leaf_partitions,
            num_partitions=num_partitions,
            pendant_lengths=pendant_lengths,
            leaf_weights=np.ones(num_leaves),
            trait_leaves=trait_leaves,
            trait_carriers=trait_carriers,
        )
        neighborhoods = tgs_select_calibration_neighborhoods(
            partition_coords=partition_coords,
            neighborhood_fractions=calibrate_neighborhood_fractions,
            counts=counts,
            neutral_traits=neutral_traits,
            trait_names=trait_names,
        )
        fit_kwargs["calibration_neighborhoods"] = neighborhoods
        for trait, name in enumerate(trait_names):
            __, factors = tgs_calc_calibrated_pendant_lengths(
                counts=counts,
                trait=trait,
                neutral_traits=neutral_traits,
                calibration_neighborhoods=neighborhoods,
            )
            factors = factors[counts.num_leaves[trait] > 0]
            if len(factors):
                logging.info(
                    "- alifestd_tgs_pendantmdc_estimate_asexual: "
                    f"calibrated trait {name}, rescaling pendant lengths by "
                    f"{factors.min():.3g} to {factors.max():.3g} (median "
                    f"{np.median(factors):.3g})",
                )

    logging.info(
        "- alifestd_tgs_pendantmdc_estimate_asexual: "
        f"fitting {len(trait_leaves)} traits to {num_leaves} leaves...",
    )
    point = tgs_fit_traits(weights=np.ones(num_leaves), **fit_kwargs)
    res[list(point.columns)] = point
    if not num_bootstrap:
        return None

    logging.info(
        "- alifestd_tgs_pendantmdc_estimate_asexual: "
        f"fitting {num_bootstrap} bootstrap replicates...",
    )
    return tgs_bootstrap(
        fit=functools.partial(tgs_fit_traits, **fit_kwargs),
        num_items=num_leaves,
        num_bootstrap=num_bootstrap,
        seed=seed,
        parallel_backend=parallel_backend,
        progress_wrap=progress_wrap,
    )
