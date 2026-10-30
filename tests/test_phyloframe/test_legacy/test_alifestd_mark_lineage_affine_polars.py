import typing

import numpy as np
import polars as pl
import pytest

from phyloframe.legacy import (
    alifestd_mark_lineage_affine_asexual,
    alifestd_mark_lineage_affine_polars,
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
            "m": [9.0, 2.0, 3.0, 0.5, 9.0],
            "a": [1, 1, 1, 2, 5],
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
        alifestd_mark_lineage_affine_polars(apply(_make_df()), "m", "a")
        .lazy()
        .collect()
    )
    np.testing.assert_allclose(
        res["lineage_affine"].to_numpy(), [1.0, 3.0, 10.0, 2.5, 5.0]
    )


def test_expressions():
    res = alifestd_mark_lineage_affine_polars(
        _make_df(), pl.lit(1.0), pl.col("a") * 2
    )
    np.testing.assert_allclose(
        res["lineage_affine"].to_numpy(), [2.0, 4.0, 6.0, 6.0, 10.0]
    )


def test_matches_pandas():
    df = _make_df()
    expected = alifestd_mark_lineage_affine_asexual(df.to_pandas(), "m", "a")
    res = alifestd_mark_lineage_affine_polars(df, "m", "a")
    np.testing.assert_allclose(
        res["lineage_affine"].to_numpy(), expected["lineage_affine"]
    )


def test_mark_as():
    res = alifestd_mark_lineage_affine_polars(
        _make_df(), "m", "a", mark_as="blueberry"
    )
    assert "blueberry" in res.columns
    assert "lineage_affine" not in res.columns


@pytest.mark.parametrize("missing", [None, float("nan")])
def test_missing_values_not_implemented(missing: typing.Optional[float]):
    df = _make_df().with_columns(
        pl.when(pl.col("id") == 1)
        .then(pl.lit(missing, dtype=pl.Float64))
        .otherwise(pl.col("a"))
        .alias("a"),
    )
    with pytest.raises(NotImplementedError):
        alifestd_mark_lineage_affine_polars(df, "m", "a")


def test_empty():
    df = pl.DataFrame(
        {"id": [], "ancestor_id": [], "m": [], "a": []},
        schema={
            "id": pl.Int64,
            "ancestor_id": pl.Int64,
            "m": pl.Float64,
            "a": pl.Float64,
        },
    )
    res = alifestd_mark_lineage_affine_polars(df, "m", "a")
    assert "lineage_affine" in res.columns
    assert len(res) == 0


def test_missing_column():
    with pytest.raises(pl.exceptions.ColumnNotFoundError):
        alifestd_mark_lineage_affine_polars(_make_df(), "nope", "a")


def test_non_contiguous_ids():
    df = pl.DataFrame(
        {"id": [0, 2], "ancestor_id": [0, 0], "m": [1.0, 1.0], "a": [1.0, 1.0]}
    )
    with pytest.raises(NotImplementedError):
        alifestd_mark_lineage_affine_polars(df, "m", "a")


def test_integer_dtype_preserved():
    df = _make_df().with_columns(pl.Series("m", [9, 2, 3, 1, 9]))
    res = alifestd_mark_lineage_affine_polars(df, "m", "a")
    assert res["lineage_affine"].dtype.is_integer()
    assert res["lineage_affine"].to_list() == [1, 3, 10, 3, 5]
