import typing

import polars as pl
import pytest

from phyloframe.legacy._alifestd_mark_oldest_root_polars import (
    alifestd_mark_oldest_root_polars as alifestd_mark_oldest_root_polars_,
)

from ._impl import enforce_dtype_stability_polars

alifestd_mark_oldest_root_polars = enforce_dtype_stability_polars(
    alifestd_mark_oldest_root_polars_,
)


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_single_root(apply: typing.Callable):
    df = apply(pl.DataFrame({"id": [0, 1, 2], "ancestor_id": [0, 0, 0]}))
    result = alifestd_mark_oldest_root_polars(df).lazy().collect()
    assert "is_oldest_root" in result.columns
    assert result["is_oldest_root"].to_list() == [True, False, False]


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_single_node(apply: typing.Callable):
    df = apply(pl.DataFrame({"id": [0], "ancestor_id": [0]}))
    result = alifestd_mark_oldest_root_polars(df).lazy().collect()
    assert result["is_oldest_root"].to_list() == [True]


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_multiple_roots_with_origin_time(apply: typing.Callable):
    df = apply(
        pl.DataFrame(
            {
                "id": [0, 1, 2],
                "ancestor_id": [0, 1, 2],
                "origin_time": [5.0, 1.0, 3.0],
            }
        )
    )
    result = alifestd_mark_oldest_root_polars(df).lazy().collect()
    # Node 1 has lowest origin_time (1.0)
    assert result["is_oldest_root"].to_list() == [False, True, False]


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
        )
    )
    result = alifestd_mark_oldest_root_polars(df).lazy().collect()
    assert "is_oldest_root" in result.columns
    assert result.is_empty()
