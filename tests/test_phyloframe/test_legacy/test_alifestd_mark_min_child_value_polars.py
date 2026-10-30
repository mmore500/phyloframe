import typing

import numpy as np
import polars as pl
import pytest

from phyloframe.legacy import (
    alifestd_mark_min_child_value_asexual,
    alifestd_mark_min_child_value_polars,
)


def _make_df() -> pl.DataFrame:
    """Tree:
    0 (root)
    +-- 1
    |   +-- 2
    +-- 3
    4 (root)
    """
    return pl.DataFrame(
        {
            "id": [0, 1, 2, 3, 4],
            "ancestor_id": [0, 0, 1, 0, 4],
            "v": [0.0, 10.0, 20.0, 30.0, 7.0],
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
        alifestd_mark_min_child_value_polars(apply(_make_df()), "v")
        .lazy()
        .collect()
    )
    assert res["min_child_value"].to_list() == [10.0, 20.0, 20.0, 30.0, 7.0]


def test_expression():
    res = alifestd_mark_min_child_value_polars(_make_df(), -pl.col("v"))
    assert res["min_child_value"].to_list() == [
        -30.0,
        -20.0,
        -20.0,
        -30.0,
        -7.0,
    ]


def test_matches_pandas():
    df = _make_df()
    expected = alifestd_mark_min_child_value_asexual(df.to_pandas(), "v")
    res = alifestd_mark_min_child_value_polars(df, "v")
    np.testing.assert_array_equal(
        res["min_child_value"].to_numpy(), expected["min_child_value"]
    )


def test_mark_as():
    res = alifestd_mark_min_child_value_polars(
        _make_df(), "v", mark_as="blueberry"
    )
    assert "blueberry" in res.columns
    assert "min_child_value" not in res.columns


@pytest.mark.parametrize("missing", [None, float("nan")])
def test_missing_values_not_implemented(missing: typing.Optional[float]):
    df = _make_df().with_columns(
        pl.when(pl.col("id") == 2)
        .then(pl.lit(missing, dtype=pl.Float64))
        .otherwise(pl.col("v"))
        .alias("v"),
    )
    with pytest.raises(NotImplementedError):
        alifestd_mark_min_child_value_polars(df, "v")


def test_empty():
    df = pl.DataFrame(
        {"id": [], "ancestor_id": [], "v": []},
        schema={"id": pl.Int64, "ancestor_id": pl.Int64, "v": pl.Float64},
    )
    res = alifestd_mark_min_child_value_polars(df, "v")
    assert "min_child_value" in res.columns
    assert len(res) == 0


def test_missing_column():
    with pytest.raises(pl.exceptions.ColumnNotFoundError):
        alifestd_mark_min_child_value_polars(_make_df(), "nope")


def test_non_contiguous_ids():
    df = pl.DataFrame({"id": [0, 2], "ancestor_id": [0, 0], "v": [1.0, 1.0]})
    with pytest.raises(NotImplementedError):
        alifestd_mark_min_child_value_polars(df, "v")


def test_integer_dtype_preserved():
    df = _make_df().with_columns(pl.col("v").cast(pl.Int64))
    res = alifestd_mark_min_child_value_polars(df, "v")
    assert res["min_child_value"].dtype.is_integer()
    assert res["min_child_value"].to_list() == [10, 20, 20, 30, 7]
