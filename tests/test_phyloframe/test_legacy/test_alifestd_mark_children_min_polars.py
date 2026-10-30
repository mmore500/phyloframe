import typing

import numpy as np
import polars as pl
import pytest

from phyloframe.legacy import (
    alifestd_mark_children_min_asexual,
    alifestd_mark_children_min_polars,
)


def _make_df() -> pl.DataFrame:
    """Tree:
    0 (root)
    +-- 1
    |   +-- 2 (v=null)
    +-- 3
    4 (root)
    """
    return pl.DataFrame(
        {
            "id": [0, 1, 2, 3, 4],
            "ancestor_id": [0, 0, 1, 0, 4],
            "v": [0.0, 10.0, None, 30.0, 7.0],
        },
    )


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_simple_tree(apply: typing.Callable):
    res = (
        alifestd_mark_children_min_polars(apply(_make_df()), "v")
        .lazy()
        .collect()
    )
    assert res["children_min"].to_list() == [10.0, None, None, None, None]


def test_expression():
    res = alifestd_mark_children_min_polars(_make_df(), -pl.col("v"))
    assert res["children_min"].to_list() == [-30.0, None, None, None, None]


def test_skipna_false():
    df = _make_df().with_columns(
        pl.when(pl.col("id") == 3).then(None).otherwise(pl.col("v")).alias("v")
    )
    res = alifestd_mark_children_min_polars(df, "v", skipna=False)
    assert res["children_min"].null_count() == len(df)


def test_matches_pandas():
    df = _make_df()
    expected = alifestd_mark_children_min_asexual(df.to_pandas(), "v")
    res = alifestd_mark_children_min_polars(df, "v")
    np.testing.assert_array_equal(
        res["children_min"].fill_null(np.nan).to_numpy(),
        expected["children_min"],
    )


def test_mark_as():
    res = alifestd_mark_children_min_polars(
        _make_df(), "v", mark_as="blueberry"
    )
    assert "blueberry" in res.columns
    assert "children_min" not in res.columns


def test_empty():
    df = pl.DataFrame(
        {"id": [], "ancestor_id": [], "v": []},
        schema={"id": pl.Int64, "ancestor_id": pl.Int64, "v": pl.Float64},
    )
    res = alifestd_mark_children_min_polars(df, "v")
    assert "children_min" in res.columns
    assert len(res) == 0


def test_missing_column():
    with pytest.raises(ValueError):
        alifestd_mark_children_min_polars(_make_df(), "nope")


def test_non_contiguous_ids():
    df = pl.DataFrame({"id": [0, 2], "ancestor_id": [0, 0], "v": [1.0, 1.0]})
    with pytest.raises(NotImplementedError):
        alifestd_mark_children_min_polars(df, "v")
