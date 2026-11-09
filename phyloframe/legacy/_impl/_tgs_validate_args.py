import typing

import numpy as np


def tgs_validate_args(
    *,
    trait_transition_cols: typing.Sequence[str],
    neutral_traits: typing.Sequence[str],
    calibrate_neighborhood_fractions: typing.Sequence[float],
    calibrate_max_rate_ratio: float,
    length_rtol: float,
    growth_exponent_bound: float,
    calibration_compare_by_vals: typing.Sequence[str],
) -> typing.Tuple[typing.List[str], typing.List[int]]:
    """Implementation detail for `alifestd_tgs_estimate_polars`. Acronym tgs
    stands for trait gain/selection.

    Validates arguments shared by the trait gain/selection estimators.
    Returns trait names, i.e., `trait_transition_cols`, and
    `neutral_traits`, given as names among them, as sorted, distinct
    indices.
    """
    trait_transition_cols = list(trait_transition_cols)
    for column in trait_transition_cols:
        if not isinstance(column, str):
            raise TypeError(
                "trait_transitions must be column names, got "
                f"{type(column).__name__}",
            )
    if len(set(trait_transition_cols)) != len(trait_transition_cols):
        raise ValueError("trait_transitions must not have duplicate columns")
    for trait in neutral_traits:
        if not isinstance(trait, str):
            raise TypeError(
                "neutral_traits must be trait_transitions column names, got "
                f"{type(trait).__name__}",
            )
        if trait not in trait_transition_cols:
            raise ValueError(
                f"neutral trait {trait!r} is not among trait_transitions",
            )
    neutral_traits = (
        np.unique(
            [trait_transition_cols.index(trait) for trait in neutral_traits]
        )
        .astype(int)
        .tolist()
    )
    if len(neutral_traits) < 2:
        raise NotImplementedError(
            "at least two distinct neutral traits are required, to calibrate "
            "lengths and as the reference for gain ratios",
        )
    if not calibrate_neighborhood_fractions or not all(
        0 < fraction <= 1 for fraction in calibrate_neighborhood_fractions
    ):
        raise ValueError(
            "calibrate_neighborhood_fractions must be nonempty, with each in "
            "(0, 1]",
        )
    if not calibrate_max_rate_ratio > 0:
        raise ValueError(
            "calibrate_max_rate_ratio must be positive, got "
            f"{calibrate_max_rate_ratio}",
        )
    if not length_rtol >= 0:
        raise ValueError(f"length_rtol must be nonnegative, got {length_rtol}")
    if not growth_exponent_bound < np.log(np.finfo(np.float64).max):
        raise ValueError("growth_exponent_bound overflows float64")
    if not len(calibration_compare_by_vals):
        raise ValueError("calibration_compare_by_vals must be nonempty")
    return trait_transition_cols, neutral_traits
