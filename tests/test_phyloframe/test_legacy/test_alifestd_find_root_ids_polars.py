import typing

import numpy as np
import polars as pl
import pytest

from phyloframe.legacy._alifestd_find_root_ids_polars import (
    alifestd_find_root_ids_polars,
)


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_simple_tree(apply: typing.Callable):
    df = apply(
        pl.DataFrame({"id": [0, 1, 2, 3, 4], "ancestor_id": [0, 0, 0, 1, 1]})
    )
    result = alifestd_find_root_ids_polars(df)
    assert list(result) == [0]


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_multiple_roots(apply: typing.Callable):
    df = apply(pl.DataFrame({"id": [0, 1, 2], "ancestor_id": [0, 1, 2]}))
    result = alifestd_find_root_ids_polars(df)
    np.testing.assert_array_equal(sorted(result), [0, 1, 2])


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
    result = alifestd_find_root_ids_polars(df)
    assert len(result) == 0
