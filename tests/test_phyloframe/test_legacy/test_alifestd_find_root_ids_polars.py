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
def test_alifestd_find_root_ids_polars_single_root(
    apply: typing.Callable,
):
    """Single root tree."""
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0, 1, 2],
                "ancestor_id": [0, 0, 1],
            }
        ),
    )

    result = alifestd_find_root_ids_polars(df_pl)

    assert isinstance(result, np.ndarray)
    assert list(result) == [0]


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_find_root_ids_polars_two_roots(
    apply: typing.Callable,
):
    """Two independent roots."""
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0, 1, 2, 3],
                "ancestor_id": [0, 1, 0, 1],
            }
        ),
    )

    result = alifestd_find_root_ids_polars(df_pl)

    assert set(result) == {0, 1}


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_find_root_ids_polars_all_roots(
    apply: typing.Callable,
):
    """All nodes are roots."""
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0, 1, 2],
                "ancestor_id": [0, 1, 2],
            }
        ),
    )

    result = alifestd_find_root_ids_polars(df_pl)

    assert set(result) == {0, 1, 2}


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_find_root_ids_polars_empty(apply: typing.Callable):
    """Empty dataframe returns empty array."""
    df_pl = apply(
        pl.DataFrame(
            {"id": [], "ancestor_id": []},
            schema={"id": pl.Int64, "ancestor_id": pl.Int64},
        ),
    )

    result = alifestd_find_root_ids_polars(df_pl)

    assert len(result) == 0
