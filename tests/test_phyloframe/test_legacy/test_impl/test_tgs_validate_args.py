import numpy as np
import pytest

from phyloframe.legacy._impl import tgs_validate_args

_KWARGS = dict(
    calibration_neighborhood_fractions=(0.1, 1.0),
    length_rtol=1e-3,
    fit_growth_exponent_bound=600.0,
    calibration_compare_by_vals=("branch_length", "origin_time"),
)


def test_valid():
    tgs_validate_args(
        trait_transitions_candidate=["a", "c"],
        trait_transitions_neutral=["d", "b"],
        **_KWARGS,
    )
    tgs_validate_args(
        trait_transitions_candidate=[],
        trait_transitions_neutral=["d", "b"],
        **_KWARGS,
    )


@pytest.mark.parametrize("trait_transitions_neutral", [(), ["b"], "bc"])
def test_too_few_neutral_traits(trait_transitions_neutral):
    with pytest.raises(NotImplementedError, match="fewer than two"):
        tgs_validate_args(
            trait_transitions_candidate=["a"],
            trait_transitions_neutral=trait_transitions_neutral,
            **_KWARGS,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(calibration_neighborhood_fractions=()),
        dict(calibration_neighborhood_fractions=(0.0,)),
        dict(calibration_neighborhood_fractions=(1.5,)),
        dict(length_rtol=-1e-3),
        dict(length_rtol=np.nan),
        dict(calibration_compare_by_vals=()),
    ],
)
def test_invalid(kwargs: dict):
    with pytest.raises(ValueError):
        tgs_validate_args(
            **{
                "trait_transitions_candidate": ["a"],
                "trait_transitions_neutral": ("b", "c"),
                **_KWARGS,
                **kwargs,
            },
        )


@pytest.mark.parametrize(
    "trait_transitions_candidate, trait_transitions_neutral",
    [
        (["a", "a"], ["b", "c"]),  # within candidate traits
        (["a"], ["b", "b"]),  # within neutral traits
        (["a", "b"], ["b", "c"]),  # between them
    ],
)
def test_duplicate_trait_transitions_raise(
    trait_transitions_candidate, trait_transitions_neutral
):
    with pytest.raises(ValueError, match="duplicate"):
        tgs_validate_args(
            trait_transitions_candidate=trait_transitions_candidate,
            trait_transitions_neutral=trait_transitions_neutral,
            **_KWARGS,
        )


@pytest.mark.parametrize("fit_growth_exponent_bound", [710.0, np.inf])
def test_growth_exponent_bound_too_large_raises(fit_growth_exponent_bound):
    with pytest.raises(ValueError, match="fit_growth_exponent_bound"):
        tgs_validate_args(
            trait_transitions_candidate=[],
            trait_transitions_neutral=["a", "b"],
            **{
                **_KWARGS,
                "fit_growth_exponent_bound": fit_growth_exponent_bound,
            },
        )
