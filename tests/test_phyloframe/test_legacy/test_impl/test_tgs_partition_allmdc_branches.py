import numpy as np
import pytest

from phyloframe.legacy._impl import tgs_partition_allmdc_branches

from ._simulate_allmdc_branches import simulate_allmdc_branches

_KWARGS = dict(
    neutral_traits=[1, 2],
    trait_names=["focal", "n1", "n2"],
    calibration_neighborhood_fractions=(0.25, 1.0),
    num_calibration_partitions=64,
    clade_size_calibration_ratio=4.0,
    min_calibration_block_neutral_gains=1,
)


def test_partition():
    branches = simulate_allmdc_branches(4_000)
    res = tgs_partition_allmdc_branches(
        branches=branches,
        calibrate=False,
        num_survival_time_bins=8,
        **_KWARGS,
    )
    # 64 partitions in each of 3 clade-size blocks: 1, 2-4, and 5-16 leaves
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


@pytest.mark.parametrize(
    "ratio, sizes",
    [
        (2.0, [1, 2, 3, 4, 5, 8, 9, 16, 17]),
        (4.0, [1, 2, 4, 5, 16, 17]),
        (1e9, [1, 2, 1000]),
    ],
)
def test_blocks(ratio: float, sizes):
    branches = simulate_allmdc_branches(4_000)
    clade_sizes = np.resize(sizes, 4_000)
    res = tgs_partition_allmdc_branches(
        branches=branches._replace(clade_sizes=clade_sizes),
        calibrate=False,
        num_survival_time_bins=8,
        **{**_KWARGS, "clade_size_calibration_ratio": ratio},
    )
    expected = {
        2.0: [0, 1, 2, 2, 3, 3, 4, 4, 5],
        4.0: [0, 1, 1, 2, 2, 3],
        1e9: [0, 1, 1],
    }[ratio]
    block_of_size = dict(zip(sizes, expected))
    blocks = np.array([block_of_size[s] for s in clade_sizes])
    # partitions never span blocks, and each block gets its own
    for partition in range(res.num_partitions):
        assert len(np.unique(blocks[res.partitions == partition])) == 1
    assert len(np.unique(res.partitions)) == res.num_partitions


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


def test_sparse_blocks_join_block_below():
    branches = simulate_allmdc_branches(4_000)
    # ratio 2 gives blocks 0 (1 leaf), 1 (2), 2 (3-4), 3 (5-8), 4 (9)
    blocks = np.ceil(np.log2(branches.clade_sizes) - 1e-9).astype(int)
    neutral_gains = np.bincount(
        blocks[np.concatenate(branches.trait_gains[1:])], minlength=5
    )
    threshold = int(neutral_gains[4]) + 1  # block 4 too sparse
    assert (neutral_gains[:4] >= threshold).all()
    kwargs = {
        **_KWARGS,
        "clade_size_calibration_ratio": 2.0,
        "min_calibration_block_neutral_gains": threshold,
    }
    res = tgs_partition_allmdc_branches(
        branches=branches, calibrate=True, num_survival_time_bins=8, **kwargs
    )
    capped = np.minimum(blocks, 3)
    for partition in range(res.num_partitions):
        assert len(np.unique(capped[res.partitions == partition])) == 1
    # clades of 9 share partitions with clades of 5-8
    assert set(res.partitions[blocks == 4]) <= set(res.partitions[blocks == 3])
    # without calibration, blocks are not capped
    uncapped = tgs_partition_allmdc_branches(
        branches=branches, calibrate=False, num_survival_time_bins=8, **kwargs
    )
    assert not set(uncapped.partitions[blocks == 4]) & set(
        uncapped.partitions[blocks == 3]
    )
