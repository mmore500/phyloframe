import numpy as np

from phyloframe.legacy._impl import tgs_partition_allmdc_branches

from ._simulate_allmdc_branches import simulate_allmdc_branches

_KWARGS = dict(
    neutral_traits=[1, 2],
    trait_names=["focal", "n1", "n2"],
    calibration_neighborhood_fractions=(0.25, 1.0),
    num_calibration_partitions=64,
    clade_size_calibration_blocks=(2, 5),
)


def test_partition():
    branches = simulate_allmdc_branches(4_000)
    res = tgs_partition_allmdc_branches(
        branches=branches,
        calibrate=False,
        num_survival_time_bins=8,
        **_KWARGS,
    )
    # 64 partitions in each of 3 clade-size blocks: 1, 2-4, and 5+ leaves
    assert res.num_partitions == 3 * 64
    assert len(res.partitions) == 4_000
    blocks = np.searchsorted([2, 5], branches.clade_sizes, side="right")
    for partition in range(res.num_partitions):
        assert len(np.unique(blocks[res.partitions == partition])) == 1
    assert res.calibration_neighborhoods is None
    # leaves all have survival time 0, filling the lower bins, which merge
    num_bins = len(res.survival_bin_times)
    assert num_bins < 8
    assert res.survival_bin_times[0] == 0.0
    assert (np.diff(res.survival_bin_times) > 0).all()
    assert (res.survival_bins[branches.clade_sizes == 1] == 0).all()
    assert res.survival_bins.max() == num_bins - 1


def test_calibrate():
    branches = simulate_allmdc_branches(4_000)
    res = tgs_partition_allmdc_branches(
        branches=branches,
        calibrate=True,
        num_survival_time_bins=8,
        **_KWARGS,
    )
    neighborhoods = res.calibration_neighborhoods
    assert neighborhoods.shape == (3 * 64, 3 * 64)
    # neighborhoods never mix clade-size blocks
    blocks = np.searchsorted([2, 5], branches.clade_sizes, side="right")
    partition_blocks = np.zeros(res.num_partitions, dtype=int)
    partition_blocks[res.partitions] = blocks
    rows, cols = neighborhoods.nonzero()
    assert (partition_blocks[rows] == partition_blocks[cols]).all()


def test_single_block():
    branches = simulate_allmdc_branches(4_000)
    res = tgs_partition_allmdc_branches(
        branches=branches,
        calibrate=False,
        num_survival_time_bins=8,
        **{**_KWARGS, "clade_size_calibration_blocks": ()},
    )
    assert res.num_partitions == 64


def test_constant_survival_time():
    branches = simulate_allmdc_branches(1_000)
    leaves = branches._replace(survival_times=np.zeros(1_000))
    res = tgs_partition_allmdc_branches(
        branches=leaves,
        calibrate=False,
        num_survival_time_bins=8,
        **_KWARGS,
    )
    assert res.survival_bin_times.tolist() == [0.0]
    assert (res.survival_bins == 0).all()
