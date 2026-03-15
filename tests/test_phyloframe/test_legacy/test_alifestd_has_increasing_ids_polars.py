import typing

import polars as pl
import pytest

from phyloframe.legacy._alifestd_has_increasing_ids_polars import (
    alifestd_has_increasing_ids_polars as alifestd_has_increasing_ids_polars_,
)

from ._impl import enforce_dtype_stability_polars

alifestd_has_increasing_ids_polars = enforce_dtype_stability_polars(
    alifestd_has_increasing_ids_polars_
)


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
    assert alifestd_has_increasing_ids_polars(df)


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_singleton_root(apply: typing.Callable):
    df = apply(pl.DataFrame({"id": [0], "ancestor_id": [0]}))
    assert alifestd_has_increasing_ids_polars(df)


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_increasing(apply: typing.Callable):
    df = apply(pl.DataFrame({"id": [0, 1, 2, 3], "ancestor_id": [0, 0, 1, 1]}))
    assert alifestd_has_increasing_ids_polars(df)


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_not_increasing(apply: typing.Callable):
    df = apply(pl.DataFrame({"id": [0, 1, 2, 3], "ancestor_id": [0, 0, 3, 3]}))
    assert not alifestd_has_increasing_ids_polars(df)


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_equal_ids(apply: typing.Callable):
    """An id equal to its ancestor_id (root) is valid."""
    df = apply(pl.DataFrame({"id": [5, 6, 7], "ancestor_id": [5, 5, 6]}))
    assert alifestd_has_increasing_ids_polars(df)


def test_missing_ancestor_id_raises():
    df = pl.DataFrame({"id": [0, 1]})
    alifestd_has_increasing_ids_polars(df)
