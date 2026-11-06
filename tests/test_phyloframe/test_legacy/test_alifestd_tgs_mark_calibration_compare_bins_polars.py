import numpy as np
import polars as pl
import pytest

from phyloframe.legacy import (
    alifestd_tgs_mark_calibration_compare_bins_polars,
)
from phyloframe.legacy._alifestd_tgs_mark_calibration_compare_bins_polars import (  # noqa: E501
    _alifestd_tgs_mark_size_partitions_fast_path,
)

_SIZES = np.array([1, 1, 2, 3, 4, 5, 8, 9, 16])


def _calc(neutral_gain_positions, merge, min_neutral_gains=1):
    return _alifestd_tgs_mark_size_partitions_fast_path(
        clade_sizes=_SIZES,
        neutral_gain_positions=np.asarray(neutral_gain_positions, dtype=int),
        ratio=2.0,
        min_neutral_gains=min_neutral_gains if merge else 0,
    )


def test_log_scale_without_merging():
    expected = [0, 0, 1, 2, 2, 3, 3, 4, 4]
    assert _calc([], merge=False).tolist() == expected


def test_sparse_size_partitions_join_below():
    # neutral gains in size partitions 0 and 1 only
    res = _calc([0, 2], merge=True)
    assert res.tolist() == [0, 0, 1, 1, 1, 1, 1, 1, 1]


def test_leaves_keep_own_size_partition():
    # no neutral gains anywhere, so all inner branches join one size
    # partition, apart from leaves
    res = _calc([], merge=True)
    assert res.tolist() == [0, 0] + [1] * (len(_SIZES) - 2)


def test_first_inner_sparse_keeps_leaves_apart():
    # neutral gains on leaves and the largest clades only
    res = _calc([0, 1, 7, 8], merge=True)
    assert res.tolist() == [0, 0] + [1] * (len(_SIZES) - 2)


def test_empty_bins_dropped():
    # ratio 1.5 leaves bin (1, 1.5] empty, which must not count as sparse
    res = _alifestd_tgs_mark_size_partitions_fast_path(
        clade_sizes=np.array([1, 1, 2, 2, 3, 3]),
        neutral_gain_positions=np.arange(6),
        ratio=1.5,
        min_neutral_gains=1,
    )
    assert res.tolist() == [0, 0, 1, 1, 2, 2]


def test_logs_clade_sizes_and_neutral_gains(caplog):
    import logging

    with caplog.at_level(logging.INFO):
        _calc([0, 2], merge=False)
    assert "size partition 1 has clade sizes 2-2" in caplog.text
    assert "1 neutral trait gains" in caplog.text


def test_enough_neutral_gains_everywhere():
    res = _calc(range(len(_SIZES)), merge=True)
    assert res.tolist() == _calc([], merge=False).tolist()


def _make_tree() -> pl.DataFrame:
    #        0
    #     1     2
    #    3 4   5 6
    #   7 8
    return pl.DataFrame(
        {
            "id": range(9),
            "ancestor_id": [0, 0, 0, 1, 1, 2, 2, 3, 3],
            "origin_time": [0.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0],
            "n1": [0, 0, 0, 1, 0, 0, 0, 0, 0],
            "n2": [0, 0, 0, 0, 0, 0, 1, 0, 0],
        },
    )


@pytest.mark.parametrize("apply", [lambda x: x, lambda x: x.lazy()])
def test_polars(apply):
    res = (
        alifestd_tgs_mark_calibration_compare_bins_polars(
            apply(_make_tree()),
            neutral_traits=["n1", "n2"],
            min_neutral_gains=0,
        )
        .lazy()
        .collect()
    )
    # clade sizes are 3, 2, 2, 1, 1, 1, 1, 1 below the root, which ends no
    # branch
    assert res["tgs_calibration_partition"].to_list() == [
        -1,
        2,
        1,
        1,
        0,
        0,
        0,
        0,
        0,
    ]
    # too few branches to split any partition, so one bin each, numbered
    # consecutively across partitions
    assert res["tgs_calibration_compare_bin"].to_list() == [
        -1,
        2,
        1,
        1,
        0,
        0,
        0,
        0,
        0,
    ]


def test_polars_merges_sparse():
    # one neutral gain, on node 3 (clade size 2), so partitions of larger
    # clades join partition 1
    res = alifestd_tgs_mark_calibration_compare_bins_polars(
        _make_tree(),
        neutral_traits=["n1"],
        min_neutral_gains=1,
        mark_partition_as="p",
        mark_bin_as="b",
    )
    assert res["p"].to_list() == [-1, 1, 1, 1, 0, 0, 0, 0, 0]
    assert res["b"].to_list() == [-1, 1, 1, 1, 0, 0, 0, 0, 0]


def _make_comb(num_leaves: int) -> pl.DataFrame:
    # a caterpillar, each inner node with one leaf and one inner child
    ancestor_ids, origin_times = [0], [0.0]
    for i in range(1, num_leaves):
        spine = 2 * (i - 1)
        ancestor_ids += [spine, spine]  # a leaf, then the next spine node
        origin_times += [i + 1.0 + 0.5 * (i % 3), float(i)]
    return pl.DataFrame(
        {
            "id": range(len(ancestor_ids)),
            "ancestor_id": ancestor_ids,
            "origin_time": origin_times,
            "n1": 0,
        },
    )


def test_polars_bins_nest_in_partitions():
    res = alifestd_tgs_mark_calibration_compare_bins_polars(
        _make_comb(200), neutral_traits=["n1"], min_neutral_gains=0, max_bins=8
    )
    marks = res.filter(pl.col("tgs_calibration_partition") >= 0)
    bins_per_partition = marks.group_by("tgs_calibration_partition").agg(
        pl.col("tgs_calibration_compare_bin").n_unique()
    )
    assert bins_per_partition["tgs_calibration_compare_bin"].max() <= 8
    assert bins_per_partition["tgs_calibration_compare_bin"].max() > 1
    # each bin lies in one partition
    assert (
        marks.group_by("tgs_calibration_compare_bin")
        .agg(pl.col("tgs_calibration_partition").n_unique())
        .select(pl.col("tgs_calibration_partition").max())
        .item()
        == 1
    )


def test_polars_empty():
    res = alifestd_tgs_mark_calibration_compare_bins_polars(
        _make_tree().clear(), neutral_traits=["n1"]
    )
    assert res["tgs_calibration_partition"].to_list() == []
    assert res["tgs_calibration_compare_bin"].to_list() == []


@pytest.mark.parametrize(
    "kwargs, error",
    [
        (dict(size_ratio=1.0), ValueError),
        (dict(min_neutral_gains=-1), ValueError),
        (dict(max_bins=6), ValueError),
        (dict(neutral_traits=["absent"]), ValueError),
        (dict(neutral_traits="n1"), TypeError),
    ],
)
def test_polars_invalid(kwargs, error):
    with pytest.raises(error):
        alifestd_tgs_mark_calibration_compare_bins_polars(
            _make_tree(), **{"neutral_traits": ["n1"], **kwargs}
        )


def test_polars_unsorted_raises():
    df = pl.DataFrame(
        {
            "id": [0, 1, 2],
            "ancestor_id": [1, 1, 1],
            "origin_time": [1.0, 0.0, 2.0],
            "n1": 0,
        },
    )
    with pytest.raises(NotImplementedError):
        alifestd_tgs_mark_calibration_compare_bins_polars(
            df, neutral_traits=["n1"]
        )
