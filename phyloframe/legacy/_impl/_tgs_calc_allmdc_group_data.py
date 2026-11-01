import typing

import numpy as np

from ._TgsAllmdcGroup import TgsAllmdcGroup
from ._tgs_calc_calibrated_pendant_lengths import (
    tgs_calc_calibrated_pendant_lengths,
)
from ._tgs_count_traits import tgs_count_traits
from ._tgs_find_reference_traits import tgs_find_reference_traits

# per partition: branch counts, gain counts, and mean lengths
TgsAllmdcGroupData = typing.Tuple[np.ndarray, np.ndarray, np.ndarray]


def tgs_calc_allmdc_group_data(
    *,
    groups: typing.Sequence[TgsAllmdcGroup],
    unit_weights: typing.Sequence[np.ndarray],
    neutral_traits: typing.Sequence[int],
    calibrate: bool,
    num_traits: int,
) -> typing.List[typing.List[typing.Optional[TgsAllmdcGroupData]]]:
    """Implementation detail for `alifestd_tgs_allmdc_estimate_asexual`.

    Here, tgs stands for trait gain/selection.

    Returns, per trait and group, partitions' (weighted) branch counts,
    gain counts, and mean lengths, omitting empty partitions. With
    `calibrate`, lengths are calibrated against neutral traits within each
    group, then rescaled so that neutral traits gain at the first
    (single-leaf) group's rate per unit length; a group without neutral
    gains to calibrate against is None.
    """
    counts = [
        tgs_count_traits(
            leaf_partitions=group.partitions,
            num_partitions=group.num_partitions,
            pendant_lengths=group.branch_lengths,
            leaf_weights=weights,
            trait_leaves=group.trait_units,
            trait_carriers=group.trait_gains,
        )
        for group, weights in zip(groups, unit_weights)
    ]
    data = [[None] * len(groups) for __ in range(num_traits)]
    for trait in range(num_traits):
        refs = tgs_find_reference_traits(
            neutral_traits=neutral_traits, trait=trait
        )
        clock = None  # first group's neutral gain rate
        for j, (group, cnt) in enumerate(zip(groups, counts)):
            lengths, factors = tgs_calc_calibrated_pendant_lengths(
                counts=cnt,
                trait=trait,
                neutral_traits=neutral_traits,
                calibration_neighborhoods=(
                    group.calibration_neighborhoods if calibrate else None
                ),
            )
            if calibrate:
                # neutral reference gain rate per calibrated unit length;
                # matching it across groups cancels distortion shared with
                # neutral traits, e.g., of inner versus pendant branches
                exposure = cnt.pendant_length_sums[refs].sum(axis=0) * factors
                with np.errstate(divide="ignore", invalid="ignore"):
                    rate = cnt.num_carriers[refs].sum() / exposure.sum()
                if not np.isfinite(rate) or rate <= 0:
                    continue  # no neutral gains here to calibrate against
                if j == 0:
                    clock = rate
                if clock is None:
                    continue
                lengths = lengths * rate / clock
            n = cnt.num_leaves[trait]
            keep = n > 0
            data[trait][j] = (
                n[keep],
                cnt.num_carriers[trait][keep],
                lengths[keep],
            )
    return data
