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


def test_names():
    names, neutral = tgs_validate_args(
        trait_transition_cols=["a", "b", "c"],
        neutral_traits=("b", "c"),
        **_KWARGS,
    )
    assert names == ["a", "b", "c"]
    assert neutral == [1, 2]


def test_non_column_raises():
    with pytest.raises(TypeError):
        tgs_validate_args(
            trait_transition_cols=["a", np.zeros(3, dtype=np.int8)],
            neutral_traits=(),
            **_KWARGS,
        )


def test_neutral_traits_normalized():
    names, neutral = tgs_validate_args(
        trait_transition_cols=["a", "b", "c", "d"],
        neutral_traits=["d", "b", "d"],
        **_KWARGS,
    )
    assert neutral == [1, 3]


def test_neutral_trait_index_raises():
    with pytest.raises(TypeError):
        tgs_validate_args(
            trait_transition_cols=["a", "b"], neutral_traits=[1], **_KWARGS
        )


@pytest.mark.parametrize("neutral_traits", [(), ["b"], ["b", "b"]])
def test_too_few_neutral_traits(neutral_traits):
    with pytest.raises(NotImplementedError, match="at least two"):
        tgs_validate_args(
            trait_transition_cols=["a", "b", "c"],
            neutral_traits=neutral_traits,
            **_KWARGS,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(neutral_traits=["b", "x"]),
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
                "trait_transition_cols": ["a", "b", "c"],
                "neutral_traits": ("b", "c"),
                **_KWARGS,
                **kwargs,
            },
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(trait_transition_cols="a", neutral_traits=()),
        dict(trait_transition_cols=["a", "b"], neutral_traits="a"),
        dict(
            trait_transition_cols=["a", "b", "c"],
            neutral_traits=["b", "c"],
            calibration_compare_by_vals="origin_time",
        ),
    ],
)
def test_bare_string_raises(kwargs: dict):
    with pytest.raises(TypeError, match="single string"):
        tgs_validate_args(**{**_KWARGS, **kwargs})


def test_duplicate_trait_transitions_raise():
    with pytest.raises(ValueError, match="duplicate"):
        tgs_validate_args(
            trait_transition_cols=["a", "b", "a"],
            neutral_traits=["a", "b"],
            **_KWARGS,
        )


@pytest.mark.parametrize("growth_exponent_bound", [710.0, np.inf])
def test_growth_exponent_bound_too_large_raises(growth_exponent_bound):
    with pytest.raises(ValueError, match="growth_exponent_bound"):
        tgs_validate_args(
            trait_transition_cols=["a", "b"],
            neutral_traits=["a", "b"],
            **{**_KWARGS, "growth_exponent_bound": growth_exponent_bound},
        )
