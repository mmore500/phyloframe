import numpy as np

from phyloframe.legacy._impl import tgs_calc_size_partitions

_SIZES = np.array([1, 1, 2, 3, 4, 5, 8, 9, 16])


def _calc(neutral_gain_positions, calibrate, min_neutral_gains=1):
    return tgs_calc_size_partitions(
        clade_sizes=_SIZES,
        neutral_gain_positions=np.asarray(neutral_gain_positions, dtype=int),
        calibrate=calibrate,
        calibrate_size_partition_ratio=2.0,
        calibrate_size_partition_min_neutral_gains=min_neutral_gains,
    )


def test_log_scale_without_calibration():
    expected = [0, 0, 1, 2, 2, 3, 3, 4, 4]
    assert _calc([], calibrate=False).tolist() == expected


def test_sparse_size_partitions_join_below():
    # neutral gains in size partitions 0 and 1 only
    res = _calc([0, 2], calibrate=True)
    assert res.tolist() == [0, 0, 1, 1, 1, 1, 1, 1, 1]


def test_leaves_keep_own_size_partition():
    # no neutral gains anywhere, so all inner branches join one size
    # partition, apart from leaves
    res = _calc([], calibrate=True)
    assert res.tolist() == [0, 0] + [1] * (len(_SIZES) - 2)


def test_first_inner_sparse_keeps_leaves_apart():
    # neutral gains on leaves and the largest clades only
    res = _calc([0, 1, 7, 8], calibrate=True)
    assert res.tolist() == [0, 0] + [1] * (len(_SIZES) - 2)


def test_empty_bins_dropped():
    # ratio 1.5 leaves bin (1, 1.5] empty, which must not count as sparse
    res = tgs_calc_size_partitions(
        clade_sizes=np.array([1, 1, 2, 2, 3, 3]),
        neutral_gain_positions=np.arange(6),
        calibrate=True,
        calibrate_size_partition_ratio=1.5,
        calibrate_size_partition_min_neutral_gains=1,
    )
    assert res.tolist() == [0, 0, 1, 1, 2, 2]


def test_logs_clade_sizes_and_neutral_gains(caplog):
    import logging

    with caplog.at_level(logging.INFO):
        _calc([0, 2], calibrate=False)
    assert "size partition 1 has clade sizes 2-2" in caplog.text
    assert "1 neutral trait gains" in caplog.text


def test_enough_neutral_gains_everywhere():
    res = _calc(range(len(_SIZES)), calibrate=True)
    assert res.tolist() == _calc([], calibrate=False).tolist()
