import numpy as np
import pytest

from phyloframe.legacy._impl import tgs_validate_args

_KWARGS = dict(
    calibrate=False,
    calibration_neighborhood_fractions=(0.1, 1.0),
    num_calibration_partitions=64,
)


def test_default_names():
    masks = ["a", np.zeros(3, dtype=bool), "c"]
    names, neutral = tgs_validate_args(
        trait_masks=masks, trait_names=None, neutral_traits=(), **_KWARGS
    )
    assert names == ["a", "1", "c"]
    assert neutral == []


def test_neutral_traits_normalized():
    names, neutral = tgs_validate_args(
        trait_masks=["a", "b", "c", "d"],
        trait_names=None,
        neutral_traits=["d", 1, -1, -3],
        **_KWARGS,
    )
    assert neutral == [1, 3]


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(trait_names=["only"]),
        dict(neutral_traits=[0], calibrate=True),
        dict(calibration_neighborhood_fractions=()),
        dict(calibration_neighborhood_fractions=(0.0,)),
        dict(calibration_neighborhood_fractions=(1.5,)),
        dict(num_calibration_partitions=48),
        dict(num_calibration_partitions=0),
    ],
)
def test_invalid(kwargs: dict):
    with pytest.raises(ValueError):
        tgs_validate_args(
            **{
                "trait_masks": ["a", "b"],
                "trait_names": None,
                "neutral_traits": (),
                **_KWARGS,
                **kwargs,
            },
        )
