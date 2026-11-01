import typing

import numpy as np

from ._TgsAllmdcGroup import TgsAllmdcGroup
from ._tgs_count_traits import tgs_count_traits
from ._tgs_partition_leaves import tgs_partition_leaves
from ._tgs_select_calibration_neighborhoods import (
    tgs_select_calibration_neighborhoods,
)


def tgs_make_allmdc_group(
    *,
    branch_lengths: np.ndarray,
    origin_times: np.ndarray,
    trait_units: typing.Sequence[np.ndarray],
    trait_gains: typing.Sequence[np.ndarray],
    neutral_traits: typing.Sequence[int],
    trait_names: typing.Sequence[str],
    calibrate: bool,
    calibration_neighborhood_fractions: typing.Sequence[float],
    num_calibration_partitions: int,
) -> typing.Optional[TgsAllmdcGroup]:
    """Implementation detail for `alifestd_tgs_allmdc_estimate_asexual`.

    Here, tgs stands for trait gain/selection.

    Partitions one clade-size group's branches by ranked length and origin
    time and, with `calibrate`, selects its calibration neighborhoods.
    Returns None if calibrating a group without neutral trait gains, which
    has nothing to calibrate against.
    """
    partitions, num_partitions, coords = tgs_partition_leaves(
        pendant_lengths=branch_lengths,
        origin_times=origin_times,
        max_partitions=num_calibration_partitions,
    )
    neighborhoods = None
    if calibrate:
        counts = tgs_count_traits(
            leaf_partitions=partitions,
            num_partitions=num_partitions,
            pendant_lengths=branch_lengths,
            leaf_weights=np.ones(len(branch_lengths)),
            trait_leaves=trait_units,
            trait_carriers=trait_gains,
        )
        if not counts.num_carriers[neutral_traits].any():
            return None
        neighborhoods = tgs_select_calibration_neighborhoods(
            partition_coords=coords,
            neighborhood_fractions=calibration_neighborhood_fractions,
            counts=counts,
            neutral_traits=neutral_traits,
            trait_names=trait_names,
        )
    return TgsAllmdcGroup(
        partitions=partitions,
        num_partitions=num_partitions,
        branch_lengths=branch_lengths,
        trait_units=list(trait_units),
        trait_gains=list(trait_gains),
        calibration_neighborhoods=neighborhoods,
    )
