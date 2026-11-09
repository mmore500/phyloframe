import numpy as np
import pytest

from phyloframe.legacy._impl import tgs_validate_args

_KWARGS = dict(
    calibrate_neighborhood_fractions=(0.1, 1.0),
    calibrate_max_rate_ratio=np.inf,
    length_rtol=1e-3,
    growth_exponent_bound=600.0,
    calibration_compare_by_vals=("branch_length", "origin_time"),
)


def test_valid():
    tgs_validate_args(
        trait_transitions=["a", "b", "c", "d"],
        neutral_traits=["d", "b", "d"],
        **_KWARGS,
    )


@pytest.mark.parametrize("neutral_traits", [(), ["b"], ["b", "b"]])
def test_too_few_neutral_traits(neutral_traits):
    with pytest.raises(NotImplementedError, match="at least two"):
        tgs_validate_args(
            trait_transitions=["a", "b", "c"],
            neutral_traits=neutral_traits,
            **_KWARGS,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(neutral_traits=["b", "x"]),
        dict(neutral_traits=[1, 2]),
        dict(calibrate_neighborhood_fractions=()),
        dict(calibrate_neighborhood_fractions=(0.0,)),
        dict(calibrate_neighborhood_fractions=(1.5,)),
        dict(calibrate_max_rate_ratio=0.0),
        dict(calibrate_max_rate_ratio=-1.0),
        dict(calibrate_max_rate_ratio=np.nan),
        dict(length_rtol=-1e-3),
        dict(length_rtol=np.nan),
        dict(calibration_compare_by_vals=()),
    ],
)
def test_invalid(kwargs: dict):
    with pytest.raises(ValueError):
        tgs_validate_args(
            **{
                "trait_transitions": ["a", "b", "c"],
                "neutral_traits": ("b", "c"),
                **_KWARGS,
                **kwargs,
            },
        )


def test_duplicate_trait_transitions_raise():
    with pytest.raises(ValueError, match="duplicate"):
        tgs_validate_args(
            trait_transitions=["a", "b", "a"],
            neutral_traits=["a", "b"],
            **_KWARGS,
        )


@pytest.mark.parametrize("growth_exponent_bound", [710.0, np.inf])
def test_growth_exponent_bound_too_large_raises(growth_exponent_bound):
    with pytest.raises(ValueError, match="growth_exponent_bound"):
        tgs_validate_args(
            trait_transitions=["a", "b"],
            neutral_traits=["a", "b"],
            **{**_KWARGS, "growth_exponent_bound": growth_exponent_bound},
        )
