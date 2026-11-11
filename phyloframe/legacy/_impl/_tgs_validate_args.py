import typing

import numpy as np


def tgs_validate_args(
    *,
    trait_transitions_candidate: typing.Sequence[str],
    trait_transitions_neutral: typing.Sequence[str],
    calibration_compare_by_vals: typing.Sequence[str],
    calibrate_neighborhood_fractions: typing.Sequence[float],
    calibrate_max_rate_ratio: float,
    length_rtol: float,
    growth_exponent_bound: float,
) -> None:
    """Implementation detail for `alifestd_tgs_estimate_polars`.

    Raises ValueError for invalid arguments, and NotImplementedError for
    fewer than two neutral traits.
    """
    if (
        isinstance(trait_transitions_neutral, str)  # one trait's name
        or len(trait_transitions_neutral) < 2
    ):
        raise NotImplementedError(
            "fewer than two neutral traits not yet supported; they calibrate "
            "lengths and are the reference for gain ratios",
        )
    trait_transitions = [
        *trait_transitions_candidate,
        *trait_transitions_neutral,
    ]
    if len(set(trait_transitions)) != len(trait_transitions):
        raise ValueError(
            "trait_transitions_candidate and trait_transitions_neutral must "
            "not have duplicate columns",
        )
    if not len(calibration_compare_by_vals):
        raise ValueError("calibration_compare_by_vals must be nonempty")
    if not len(calibrate_neighborhood_fractions) or not all(
        0 < fraction <= 1 for fraction in calibrate_neighborhood_fractions
    ):
        raise ValueError(
            "calibrate_neighborhood_fractions must be nonempty, with each in "
            "(0, 1]",
        )
    if not calibrate_max_rate_ratio > 0:
        raise ValueError("calibrate_max_rate_ratio must be positive")
    if not length_rtol >= 0:
        raise ValueError("length_rtol must be nonnegative")
    if not growth_exponent_bound < np.log(np.finfo(np.float64).max):
        raise ValueError("growth_exponent_bound overflows float64")
