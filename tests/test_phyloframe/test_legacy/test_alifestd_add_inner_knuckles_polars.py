import typing

import polars as pl
import pytest

from phyloframe.legacy._alifestd_add_inner_knuckles_polars import (
    alifestd_add_inner_knuckles_polars as alifestd_add_inner_knuckles_polars_,
)

from ._impl import enforce_dtype_stability_polars

alifestd_add_inner_knuckles_polars = enforce_dtype_stability_polars(
    alifestd_add_inner_knuckles_polars_
)


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_simple(apply: typing.Callable):
    df = apply(
        pl.DataFrame(
            {
                "id": [0, 1, 2, 3, 4],
                "ancestor_id": [0, 0, 0, 1, 1],
            }
        ),
    )

    result = alifestd_add_inner_knuckles_polars(df)
    result = result.lazy().collect().sort("id")

    # 5 original + 2 knuckles for inner nodes 0 and 1
    assert len(result) == 7

    # Original inner nodes (0, 1) should have ancestor_id pointing to knuckle
    row0 = result.filter(pl.col("id") == 0)
    assert row0["ancestor_id"].item() == 5  # knuckle id = 0 + 5

    row1 = result.filter(pl.col("id") == 1)
    assert row1["ancestor_id"].item() == 6  # knuckle id = 1 + 5

    # Knuckle of node 0 (root): id=5, ancestor_id=5 (root shifted)
    knuckle0 = result.filter(pl.col("id") == 5)
    assert knuckle0["ancestor_id"].item() == 5

    # Knuckle of node 1: id=6, ancestor_id=0 (stays same as original)
    knuckle1 = result.filter(pl.col("id") == 6)
    assert knuckle1["ancestor_id"].item() == 0


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_empty(apply: typing.Callable):
    df = apply(
        pl.DataFrame(
            {"id": [], "ancestor_id": []},
            schema={"id": pl.Int64, "ancestor_id": pl.Int64},
        ),
    )

    result = alifestd_add_inner_knuckles_polars(df).lazy().collect()
    assert result.is_empty()


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_all_leaves(apply: typing.Callable):
    df = apply(
        pl.DataFrame(
            {
                "id": [0],
                "ancestor_id": [0],
            }
        ),
    )

    result = alifestd_add_inner_knuckles_polars(df).lazy().collect()

    # Single root node with no children is a leaf; no knuckles added
    assert len(result) == 1
    assert result["id"].to_list() == [0]


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_non_contiguous_ids(apply: typing.Callable):
    df = apply(
        pl.DataFrame(
            {
                "id": [0, 2, 5],
                "ancestor_id": [0, 0, 2],
            }
        ),
    )

    with pytest.raises(NotImplementedError):
        alifestd_add_inner_knuckles_polars(df).lazy().collect()


def test_with_is_root_column():
    """Knuckles should have is_root=False when is_root column present."""
    df = pl.DataFrame(
        {
            "id": [0, 1, 2, 3, 4],
            "ancestor_id": [0, 0, 0, 1, 1],
            "is_root": [True, False, False, False, False],
        }
    )

    result = alifestd_add_inner_knuckles_polars(df).lazy().collect()
    knuckles = result.filter(pl.col("id") >= 5)
    assert knuckles["is_root"].to_list() == [False, False]


def test_with_origin_time_delta_column():
    """Knuckles should have origin_time_delta=0 when column present."""
    df = pl.DataFrame(
        {
            "id": [0, 1, 2, 3, 4],
            "ancestor_id": [0, 0, 0, 1, 1],
            "origin_time_delta": [0, 1, 1, 2, 2],
        }
    )

    result = alifestd_add_inner_knuckles_polars(df).lazy().collect()
    knuckles = result.filter(pl.col("id") >= 5)
    assert knuckles["origin_time_delta"].to_list() == [0.0, 0.0]


def test_with_ancestor_list_column():
    """ancestor_list should be recomputed after knuckle addition."""
    df = pl.DataFrame(
        {
            "id": [0, 1, 2, 3, 4],
            "ancestor_id": [0, 0, 0, 1, 1],
            "ancestor_list": [
                "[None]",
                "[0]",
                "[0]",
                "[1]",
                "[1]",
            ],
        }
    )

    result = alifestd_add_inner_knuckles_polars(df).lazy().collect()
    assert "ancestor_list" in result.columns
    # Knuckle of root (id=5) should be self-rooted
    knuckle0 = result.filter(pl.col("id") == 5)
    assert knuckle0["ancestor_list"].item() == "[none]"
