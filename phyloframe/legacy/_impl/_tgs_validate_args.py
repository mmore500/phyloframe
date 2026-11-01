import operator
import typing

import numpy as np


def tgs_validate_args(
    *,
    trait_masks: typing.Sequence[typing.Any],
    trait_names: typing.Optional[typing.Sequence[str]],
    neutral_traits: typing.Sequence[typing.Union[int, str]],
    calibrate: bool,
    calibration_neighborhood_fractions: typing.Sequence[float],
    num_calibration_partitions: int,
) -> typing.Tuple[typing.List[str], typing.List[int]]:
    """Implementation detail for
    `alifestd_tgs_pendantmdc_estimate_asexual`,
    `alifestd_tgs_allmdc_estimate_asexual`, and their polars counterparts.

    Here, tgs stands for trait gain/selection.

    Validates arguments shared by the trait gain/selection estimators.
    Returns trait names, defaulting to column names for traits given as
    str, else trait indices; and `neutral_traits`, given as indices or
    trait names, as sorted, distinct, non-negative indices.
    """
    num_traits = len(trait_masks)
    if trait_names is None:
        trait_names = [
            mask if isinstance(mask, str) else str(i)
            for i, mask in enumerate(trait_masks)
        ]
    trait_names = list(trait_names)
    if len(trait_names) != num_traits:
        raise ValueError(
            f"got {len(trait_names)} trait names for {num_traits} trait masks",
        )
    neutral_traits = [
        trait_names.index(trait) if isinstance(trait, str) else trait
        for trait in neutral_traits
    ]
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
    return trait_names, neutral_traits
