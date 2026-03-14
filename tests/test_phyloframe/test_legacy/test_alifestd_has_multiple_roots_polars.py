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
def test_alifestd_has_multiple_roots_polars_single_root(
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

    assert not alifestd_has_multiple_roots_polars(df_pl)


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_has_multiple_roots_polars_two_roots(
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

    assert alifestd_has_multiple_roots_polars(df_pl)


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_has_multiple_roots_polars_all_roots(
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

    assert alifestd_has_multiple_roots_polars(df_pl)


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_has_multiple_roots_polars_single_node(
    apply: typing.Callable,
):
    """A single node is not multiple roots."""
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0],
                "ancestor_id": [0],
            }
        ),
    )

    assert not alifestd_has_multiple_roots_polars(df_pl)


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_has_multiple_roots_polars_empty(
    apply: typing.Callable,
):
    """Empty dataframe has no roots."""
    df_pl = apply(
        pl.DataFrame(
            {"id": [], "ancestor_id": []},
            schema={"id": pl.Int64, "ancestor_id": pl.Int64},
        ),
    )

    assert not alifestd_has_multiple_roots_polars(df_pl)
