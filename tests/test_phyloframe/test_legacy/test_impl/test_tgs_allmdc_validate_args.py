import pytest

from phyloframe.legacy._impl import tgs_allmdc_validate_args

_VALID = dict(
    calibrate_size_partition_ratio=2.0,
    calibrate_size_partition_min_neutral_gains=1,
)


def test_valid():
    tgs_allmdc_validate_args(**_VALID)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(calibrate_size_partition_ratio=1.0),
        dict(calibrate_size_partition_min_neutral_gains=0),
    ],
)
def test_invalid(kwargs: dict):
    with pytest.raises(ValueError):
        tgs_allmdc_validate_args(**{**_VALID, **kwargs})
