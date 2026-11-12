import typing

import numpy as np


def tgs_validate_args(
    *,
    trait_transitions_candidate: typing.Sequence[str],
    trait_transitions_neutral: typing.Sequence[str],
    calibration_compare_by_bins: typing.Optional[str],
    calibration_compare_by_vals: typing.Optional[typing.Sequence[str]],
    calibration_neighborhood_candidates: typing.Sequence[float],
    nll_coarsegrain_rtol: float,
    fit_growth_exponent_bound: float,
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
            "episode durations and are the reference for gain ratios",
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
    if (calibration_compare_by_bins is None) != (
        calibration_compare_by_vals is None
    ):
        raise ValueError(
            "calibration_compare_by_bins and calibration_compare_by_vals "
            "must be given together",
        )
    if calibration_compare_by_vals is not None and not len(
        calibration_compare_by_vals
    ):
        raise ValueError("calibration_compare_by_vals must be nonempty")
    if not len(calibration_neighborhood_candidates) or not all(
        0 < fraction <= 1 for fraction in calibration_neighborhood_candidates
    ):
        raise ValueError(
            "calibration_neighborhood_candidates must be nonempty, with each "
            "in (0, 1]",
        )
    if not nll_coarsegrain_rtol >= 0:
        raise ValueError("nll_coarsegrain_rtol must be nonnegative")
    if not fit_growth_exponent_bound < np.log(np.finfo(np.float64).max):
        raise ValueError("fit_growth_exponent_bound overflows float64")
