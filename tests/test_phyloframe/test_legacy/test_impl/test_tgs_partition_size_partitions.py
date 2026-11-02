import numpy as np

from phyloframe.legacy._impl import tgs_partition_size_partitions


def test_partitions_never_mix_size_partitions():
    rng = np.random.default_rng(1)
    size_partitions = rng.integers(0, 3, 3_000)
    partitions, blocks = tgs_partition_size_partitions(
        size_partitions=size_partitions,
        branch_lengths=rng.random(3_000),
        origin_times=rng.random(3_000),
        max_partitions=16,
    )
    assert len(blocks) == 3
    offsets = [offset for offset, __, __ in blocks]
    nums = [num for __, num, __ in blocks]
    assert offsets == [0, nums[0], nums[0] + nums[1]]
    assert partitions.max() + 1 == sum(nums)
    for size_partition, (offset, num, coords) in enumerate(blocks):
        members = partitions[size_partitions == size_partition]
        assert set(members) == set(range(offset, offset + num))
        assert len(coords) == num


def test_constant_coordinate_skipped():
    rng = np.random.default_rng(1)
    partitions, [(__, num, coords)] = tgs_partition_size_partitions(
        size_partitions=np.zeros(256, dtype=int),
        branch_lengths=rng.random(256),
        origin_times=np.zeros(256),
        max_partitions=16,
    )
    assert num == 16
    assert coords.shape == (16, 1)
