import numpy as np
import pytest

from phyloframe.legacy._impl import tgs_validate_args

_KWARGS = dict(
    calibrate=False,
    calibrate_neighborhood_fractions=(0.1, 1.0),
    calibrate_num_partitions=64,
)


def test_default_names():
    names, neutral = tgs_validate_args(
        trait_transitions=["a", "b", "c"],
        neutral_traits=(),
        **_KWARGS,
    )
    assert names == ["a", "b", "c"]
    assert neutral == []


def test_non_column_raises():
    with pytest.raises(TypeError):
        tgs_validate_args(
            trait_transitions=["a", np.zeros(3, dtype=np.int8)],
            neutral_traits=(),
            **_KWARGS,
        )


def test_neutral_traits_normalized():
    names, neutral = tgs_validate_args(
        trait_transitions=["a", "b", "c", "d"],
        neutral_traits=["d", "b", "d"],
        **_KWARGS,
    )
    assert neutral == [1, 3]


def test_neutral_trait_index_raises():
    with pytest.raises(TypeError):
        tgs_validate_args(
            trait_transitions=["a", "b"], neutral_traits=[1], **_KWARGS
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(neutral_traits=["x"]),
        dict(neutral_traits=["a"], calibrate=True),
        dict(calibrate_neighborhood_fractions=()),
        dict(calibrate_neighborhood_fractions=(0.0,)),
        dict(calibrate_neighborhood_fractions=(1.5,)),
        dict(calibrate_num_partitions=48),
        dict(calibrate_num_partitions=0),
    ],
)
def test_invalid(kwargs: dict):
    with pytest.raises(ValueError):
        tgs_validate_args(
            **{
                "trait_transitions": ["a", "b"],
                "neutral_traits": (),
                **_KWARGS,
                **kwargs,
            },
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(trait_transitions="a", neutral_traits=()),
        dict(trait_transitions=["a", "b"], neutral_traits="a"),
    ],
)
def test_bare_string_raises(kwargs: dict):
    with pytest.raises(TypeError, match="single string"):
        tgs_validate_args(**kwargs, **_KWARGS)
