import operator
import typing

import numpy as np


def tgs_validate_args(
    *,
    trait_transitions: typing.Sequence[str],
    trait_names: typing.Optional[typing.Sequence[str]],
    neutral_traits: typing.Sequence[typing.Union[int, str]],
    calibrate: bool,
    calibrate_neighborhood_fractions: typing.Sequence[float],
    calibrate_num_partitions: int,
) -> typing.Tuple[typing.List[str], typing.List[int]]:
    """Implementation detail for
    `alifestd_tgs_pendantmdc_estimate_asexual`,
    `alifestd_tgs_allmdc_estimate_asexual`, and their polars counterparts.

    Here, tgs stands for trait gain/selection.

    Validates arguments shared by the trait gain/selection estimators.
    Returns trait names, defaulting to the transition column names; and
    `neutral_traits`, given as indices or trait names, as sorted, distinct,
    non-negative indices.
    """
    trait_transitions = list(trait_transitions)
    for column in trait_transitions:
        if not isinstance(column, str):
            raise TypeError(
                "trait_transitions must be column names, got "
                f"{type(column).__name__}",
            )
    num_traits = len(trait_transitions)
    trait_names = list(
        trait_transitions if trait_names is None else trait_names
    )
    if len(trait_names) != num_traits:
        raise ValueError(
            f"got {len(trait_names)} trait names for {num_traits} traits",
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
    if not calibrate_neighborhood_fractions or not all(
        0 < fraction <= 1 for fraction in calibrate_neighborhood_fractions
    ):
        raise ValueError(
            "calibrate_neighborhood_fractions must be nonempty, with each in "
            "(0, 1]",
        )
    if int(calibrate_num_partitions).bit_count() != 1 or (
        calibrate_num_partitions < 1
    ):
        raise ValueError(
            "calibrate_num_partitions must be a power of two, got "
            f"{calibrate_num_partitions}",
        )
    return trait_names, neutral_traits
