import numpy as np
import polars as pl
import polars.testing as pltest
import pytest

from phyloframe.legacy import alifestd_mark_clade_size_geom_classes_polars

_SIZES = np.array([1, 1, 2, 3, 4, 5, 8, 9, 16])


def _calc(weighted_positions, min_class_weight=1.0, sizes=_SIZES, **kwargs):
    weights = np.zeros(len(sizes))
    weights[np.asarray(weighted_positions, dtype=int)] = 1.0
    return alifestd_mark_clade_size_geom_classes_polars(
        pl.DataFrame(
            {"id": range(len(sizes)), "num_leaves": sizes, "weight": weights}
        ),
        weight_by="weight",
        min_class_weight=min_class_weight,
        **kwargs,
    )["clade_size_geom_class"]


def test_log_scale_without_merging():
    expected = [0, 0, 1, 2, 2, 3, 3, 4, 4]
    assert _calc([], min_class_weight=0.0).to_list() == expected


def test_light_classes_join_below():
    # weight in classes 0, 1, and 2 only
    res = _calc([0, 2, 3])
    assert res.to_list() == [0, 0, 1, 2, 2, 2, 2, 2, 2]


def test_class_zero_kept_apart():
    # no weight anywhere, so all classes but 0 form one class
    res = _calc([])
    assert res.to_list() == [0, 0] + [1] * (len(_SIZES) - 2)


def test_first_inner_light_keeps_class_zero_apart():
    # weight in classes 0 and 4 only
    res = _calc([0, 1, 7, 8])
    assert res.to_list() == [0, 0] + [1] * (len(_SIZES) - 2)


def test_heavy_enough_everywhere():
    res = _calc(range(len(_SIZES)))
    assert res.to_list() == _calc([], min_class_weight=0.0).to_list()


def test_empty_bins_skipped():
    # ratio 1.5 leaves bin (1, 1.5] empty, which must not count as light
    res = _calc(range(6), sizes=np.array([1, 1, 2, 2, 3, 3]), size_ratio=1.5)
    assert res.to_list() == [0, 0, 1, 1, 2, 2]


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
        alifestd_mark_clade_size_geom_classes_polars(apply(_make_tree()))
        .lazy()
        .collect()
    )
    # clade sizes are 5, 3, 2, 2, 1, 1, 1, 1, 1
    assert res["clade_size_geom_class"].to_list() == [
        3,
        2,
        1,
        1,
        0,
        0,
        0,
        0,
        0,
    ]


def test_polars_merges_light():
    # weight 1 only on node 3 (clade size 2), so classes of larger clades
    # join class 1
    res = alifestd_mark_clade_size_geom_classes_polars(
        _make_tree(), weight_by=["n1"], min_class_weight=1, mark_as="m"
    )
    assert res["m"].to_list() == [1, 1, 1, 1, 0, 0, 0, 0, 0]
    # weights sum across columns; node 6 is a leaf, in class 0
    res = alifestd_mark_clade_size_geom_classes_polars(
        _make_tree(), weight_by=["n1", "n2"], min_class_weight=1
    )
    assert res["clade_size_geom_class"].to_list() == [
        1,
        1,
        1,
        1,
        0,
        0,
        0,
        0,
        0,
    ]


def test_polars_empty():
    res = alifestd_mark_clade_size_geom_classes_polars(_make_tree().clear())
    assert res["clade_size_geom_class"].to_list() == []


@pytest.mark.parametrize(
    "kwargs, error",
    [
        (dict(size_ratio=1.0), ValueError),
        (dict(weight_by=["absent"]), pl.exceptions.ColumnNotFoundError),
    ],
)
def test_polars_invalid(kwargs, error):
    with pytest.raises(error):
        alifestd_mark_clade_size_geom_classes_polars(_make_tree(), **kwargs)


def test_polars_unsorted_raises():
    df = pl.DataFrame({"id": [0, 1, 2], "ancestor_id": [1, 1, 1]})
    with pytest.raises(NotImplementedError):
        alifestd_mark_clade_size_geom_classes_polars(df)


def test_polars_single_string_weight_by():
    pltest.assert_frame_equal(
        alifestd_mark_clade_size_geom_classes_polars(
            _make_tree(), weight_by="n1"
        ),
        alifestd_mark_clade_size_geom_classes_polars(
            _make_tree(), weight_by=["n1"]
        ),
    )
