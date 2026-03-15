import typing

import polars as pl
import pytest

from phyloframe.legacy._alifestd_mark_root_id_polars import (
    alifestd_mark_root_id_polars as alifestd_mark_root_id_polars_,
)

from ._impl import enforce_dtype_stability_polars

alifestd_mark_root_id_polars = enforce_dtype_stability_polars(
    alifestd_mark_root_id_polars_,
)


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_simple_tree(apply: typing.Callable):
    """Tree: 0(root) -> 1 -> 3, 0 -> 2"""
    df = apply(pl.DataFrame({"id": [0, 1, 2, 3], "ancestor_id": [0, 0, 0, 1]}))
    result = alifestd_mark_root_id_polars(df).lazy().collect()
    assert "root_id" in result.columns
    assert result["root_id"].to_list() == [0, 0, 0, 0]


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_single_node(apply: typing.Callable):
    df = apply(pl.DataFrame({"id": [0], "ancestor_id": [0]}))
    result = alifestd_mark_root_id_polars(df).lazy().collect()
    assert result["root_id"].to_list() == [0]


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_multiple_roots(apply: typing.Callable):
    """Two separate trees: 0->1, 2->3"""
    df = apply(pl.DataFrame({"id": [0, 1, 2, 3], "ancestor_id": [0, 0, 2, 2]}))
    result = alifestd_mark_root_id_polars(df).lazy().collect()
    assert result["root_id"].to_list() == [0, 0, 2, 2]


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
    result = alifestd_mark_root_id_polars(df).lazy().collect()
    assert "root_id" in result.columns
    assert result.is_empty()


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_non_contiguous_ids(apply: typing.Callable):
    df = apply(pl.DataFrame({"id": [0, 2, 5], "ancestor_id": [0, 0, 2]}))
    with pytest.raises(NotImplementedError):
        alifestd_mark_root_id_polars(df).lazy().collect()
