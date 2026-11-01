import numpy as np

from phyloframe.legacy._impl import tgs_make_allmdc_group

_KWARGS = dict(
    trait_names=["focal", "n1", "n2"],
    neutral_traits=[1, 2],
    calibration_neighborhood_fractions=(0.5, 1.0),
    num_calibration_partitions=8,
)


def test_uncalibrated():
    group = tgs_make_allmdc_group(
        branch_lengths=np.arange(1.0, 101.0),
        origin_times=np.arange(100.0),
        trait_units=[np.arange(100)] * 3,
        trait_gains=[np.arange(5)] * 3,
        calibrate=False,
        **_KWARGS,
    )
    assert group.num_partitions == 8
    assert len(group.partitions) == 100
    assert group.calibration_neighborhoods is None


def test_calibrated():
    group = tgs_make_allmdc_group(
        branch_lengths=np.arange(1.0, 101.0),
        origin_times=np.arange(100.0),
        trait_units=[np.arange(100)] * 3,
        trait_gains=[np.arange(0, 100, 10)] * 3,
        calibrate=True,
        **_KWARGS,
    )
    assert group.calibration_neighborhoods.shape == (8, 8)


def test_calibrated_without_neutral_gains():
    group = tgs_make_allmdc_group(
        branch_lengths=np.arange(1.0, 101.0),
        origin_times=np.arange(100.0),
        trait_units=[np.arange(100)] * 3,
        trait_gains=[np.arange(5), np.arange(0), np.arange(0)],
        calibrate=True,
        **_KWARGS,
    )
    assert group is None
