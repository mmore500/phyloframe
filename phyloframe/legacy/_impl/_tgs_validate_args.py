import operator
import typing

import numpy as np


def tgs_validate_args(
    *,
    num_traits: int,
    neutral_traits: typing.Sequence[int],
    calibrate: bool,
    calibration_neighborhood_fractions: typing.Sequence[float],
    num_calibration_partitions: int,
) -> typing.List[int]:
    """Implementation detail for
    `alifestd_estimate_trait_gain_selection_pendantmdc_asexual` and
    `alifestd_estimate_trait_gain_selection_allmdc_asexual`.

    Here, tgs stands for trait gain/selection.

    Validates arguments shared by the trait gain/selection estimators, and
    returns `neutral_traits` as sorted, distinct, non-negative indices.
    """
    neutral_traits = np.unique(
        np.arange(num_traits)[list(map(operator.index, neutral_traits))]
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
    return neutral_traits
