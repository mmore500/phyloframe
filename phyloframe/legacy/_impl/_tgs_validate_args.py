import typing

import numpy as np


def tgs_validate_args(
    *,
    trait_transitions: typing.Sequence[str],
    neutral_traits: typing.Sequence[str],
    calibrate_neighborhood_fractions: typing.Sequence[float],
    calibrate_max_rate_ratio: float,
    length_rtol: float,
) -> typing.Tuple[typing.List[str], typing.List[int]]:
    """Implementation detail for `alifestd_tgs_pendantmdc_estimate_polars` and
    `alifestd_tgs_allmdc_estimate_polars`. Acronym tgs stands for trait
    gain/selection.

    Validates arguments shared by the trait gain/selection estimators.
    Returns trait names, i.e., the transition column names, and
    `neutral_traits`, given as names among them, as sorted, distinct
    indices.
    """
    for name, value in (
        ("trait_transitions", trait_transitions),
        ("neutral_traits", neutral_traits),
    ):
        if isinstance(value, str):
            raise TypeError(
                f"{name} must be a sequence of column names, not a single "
                f"string; did you mean [{value!r}]?",
            )
    trait_transitions = list(trait_transitions)
    for column in trait_transitions:
        if not isinstance(column, str):
            raise TypeError(
                "trait_transitions must be column names, got "
                f"{type(column).__name__}",
            )
    for trait in neutral_traits:
        if not isinstance(trait, str):
            raise TypeError(
                "neutral_traits must be trait_transitions column names, got "
                f"{type(trait).__name__}",
            )
        if trait not in trait_transitions:
            raise ValueError(
                f"neutral trait {trait!r} is not among trait_transitions",
            )
    neutral_traits = (
        np.unique([trait_transitions.index(trait) for trait in neutral_traits])
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
    return trait_transitions, neutral_traits
