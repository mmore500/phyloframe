import logging

import numpy as np
import polars as pl
import pytest

from phyloframe.legacy import alifestd_mark_clade_size_classes_polars
from phyloframe.legacy._alifestd_mark_clade_size_classes_polars import (
    _alifestd_mark_clade_size_classes_fast_path,
)

_SIZES = np.array([1, 1, 2, 3, 4, 5, 8, 9, 16])


def _calc(weighted_positions, min_class_weight=1.0):
    weights = np.zeros(len(_SIZES))
    weights[np.asarray(weighted_positions, dtype=int)] = 1.0
    return _alifestd_mark_clade_size_classes_fast_path(
        clade_sizes=_SIZES,
        weights=weights,
        size_ratio=2.0,
        min_class_weight=min_class_weight,
    )


def test_log_scale_without_merging():
    expected = [0, 0, 1, 2, 2, 3, 3, 4, 4]
    assert _calc([], min_class_weight=0.0).tolist() == expected


def test_light_classes_join_below():
    # weight in classes 0 and 1 only
    res = _calc([0, 2])
    assert res.tolist() == [0, 0, 1, 1, 1, 1, 1, 1, 1]


def test_class_zero_kept_apart():
    # no weight anywhere, so all classes but 0 form one class
    res = _calc([])
    assert res.tolist() == [0, 0] + [1] * (len(_SIZES) - 2)


def test_first_inner_light_keeps_class_zero_apart():
    # weight in classes 0 and 4 only
    res = _calc([0, 1, 7, 8])
    assert res.tolist() == [0, 0] + [1] * (len(_SIZES) - 2)


def test_heavy_enough_everywhere():
    res = _calc(range(len(_SIZES)))
    assert res.tolist() == _calc([], min_class_weight=0.0).tolist()


def test_empty_bins_skipped():
    # ratio 1.5 leaves bin (1, 1.5] empty, which must not count as light
    res = _alifestd_mark_clade_size_classes_fast_path(
        clade_sizes=np.array([1, 1, 2, 2, 3, 3]),
        weights=np.ones(6),
        size_ratio=1.5,
        min_class_weight=1.0,
    )
    assert res.tolist() == [0, 0, 1, 1, 2, 2]


def test_logs_clade_sizes_and_weights(caplog):
    with caplog.at_level(logging.INFO):
        _calc([0, 2], min_class_weight=0.0)
    assert "class 1 has clade sizes 2-2, 1 nodes, and weight 1" in caplog.text


def _make_tree() -> pl.DataFrame:
    #        0
    #     1     2
    #    3 4   5 6
    #   7 8
    return pl.DataFrame(
        {
            "id": range(9),
            "ancestor_id": [0, 0, 0, 1, 1, 2, 2, 3, 3],
            "n1": [0, 0, 0, 1, 0, 0, 0, 0, 0],
            "n2": [0, 0, 0, 0, 0, 0, 1, 0, 0],
        },
    )


@pytest.mark.parametrize("apply", [lambda x: x, lambda x: x.lazy()])
def test_polars(apply):
    res = (
        alifestd_mark_clade_size_classes_polars(apply(_make_tree()))
        .lazy()
        .collect()
    )
    # clade sizes are 5, 3, 2, 2, 1, 1, 1, 1, 1
    assert res["clade_size_class"].to_list() == [3, 2, 1, 1, 0, 0, 0, 0, 0]


def test_polars_merges_light():
    # weight 1 only on node 3 (clade size 2), so classes of larger clades
    # join class 1
    res = alifestd_mark_clade_size_classes_polars(
        _make_tree(), weight_by=["n1"], min_class_weight=1, mark_as="m"
    )
    assert res["m"].to_list() == [1, 1, 1, 1, 0, 0, 0, 0, 0]
    # weights sum across columns; node 6 is a leaf, in class 0
    res = alifestd_mark_clade_size_classes_polars(
        _make_tree(), weight_by=["n1", "n2"], min_class_weight=1
    )
    assert res["clade_size_class"].to_list() == [1, 1, 1, 1, 0, 0, 0, 0, 0]


def test_polars_empty():
    res = alifestd_mark_clade_size_classes_polars(_make_tree().clear())
    assert res["clade_size_class"].to_list() == []


@pytest.mark.parametrize(
    "kwargs, error",
    [
        (dict(size_ratio=1.0), ValueError),
        (dict(min_class_weight=-1), ValueError),
        (dict(weight_by=["absent"]), ValueError),
        (dict(weight_by="n1"), TypeError),
    ],
)
def test_polars_invalid(kwargs, error):
    with pytest.raises(error):
        alifestd_mark_clade_size_classes_polars(_make_tree(), **kwargs)


def test_polars_null_weights_raise():
    df = _make_tree().with_columns(
        n1=pl.when(pl.col("id") == 2).then(None).otherwise(pl.col("n1"))
    )
    with pytest.raises(NotImplementedError, match="null"):
        alifestd_mark_clade_size_classes_polars(df, weight_by=["n1"])


def test_polars_unsorted_raises():
    df = pl.DataFrame({"id": [0, 1, 2], "ancestor_id": [1, 1, 1]})
    with pytest.raises(NotImplementedError):
        alifestd_mark_clade_size_classes_polars(df)
