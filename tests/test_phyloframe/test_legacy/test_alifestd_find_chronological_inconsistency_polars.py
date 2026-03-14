import typing

import polars as pl
import pytest

from phyloframe.legacy._alifestd_find_chronological_inconsistency_polars import (
    alifestd_find_chronological_inconsistency_polars,
)


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_find_chronological_inconsistency_polars_consistent(
    apply: typing.Callable,
):
    """No inconsistency in properly ordered tree."""
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0, 1, 2],
                "ancestor_id": [0, 0, 1],
                "origin_time": [0.0, 1.0, 2.0],
            }
        ),
    )

    assert alifestd_find_chronological_inconsistency_polars(df_pl) is None


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_find_chronological_inconsistency_polars_inconsistent(
    apply: typing.Callable,
):
    """Find inconsistency where child has earlier origin_time."""
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0, 1, 2],
                "ancestor_id": [0, 0, 1],
                "origin_time": [5.0, 1.0, 0.0],
            }
        ),
    )

    result = alifestd_find_chronological_inconsistency_polars(df_pl)
    assert result is not None


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_find_chronological_inconsistency_polars_empty(
    apply: typing.Callable,
):
    """Empty dataframe has no inconsistency."""
    df_pl = apply(
        pl.DataFrame(
            {"id": [], "ancestor_id": [], "origin_time": []},
            schema={
                "id": pl.Int64,
                "ancestor_id": pl.Int64,
                "origin_time": pl.Float64,
            },
        ),
    )

    assert alifestd_find_chronological_inconsistency_polars(df_pl) is None


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_find_chronological_inconsistency_polars_single_node(
    apply: typing.Callable,
):
    """Single node has no inconsistency."""
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0],
                "ancestor_id": [0],
                "origin_time": [0.0],
            }
        ),
    )

    assert alifestd_find_chronological_inconsistency_polars(df_pl) is None
