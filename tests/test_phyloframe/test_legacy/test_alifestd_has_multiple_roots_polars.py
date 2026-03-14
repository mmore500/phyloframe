import typing

import polars as pl
import pytest

from phyloframe.legacy._alifestd_has_multiple_roots_polars import (
    alifestd_has_multiple_roots_polars,
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
    assert not alifestd_has_multiple_roots_polars(df)


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_multiple_roots(apply: typing.Callable):
    df = apply(pl.DataFrame({"id": [0, 1, 2], "ancestor_id": [0, 1, 1]}))
    assert alifestd_has_multiple_roots_polars(df)


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
    assert not alifestd_has_multiple_roots_polars(df)
