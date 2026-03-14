import typing

import polars as pl
import pytest

from phyloframe.legacy._alifestd_chronological_sort_polars import (
    alifestd_chronological_sort_polars,
)


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_chronological_sort_polars_basic(
    apply: typing.Callable,
):
    """Test basic sorting by origin_time."""
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0, 1, 2],
                "ancestor_id": [0, 0, 1],
                "origin_time": [3.0, 1.0, 2.0],
            }
        ),
    )

    result = alifestd_chronological_sort_polars(df_pl)

    assert result["origin_time"].to_list() == [1.0, 2.0, 3.0]


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_chronological_sort_polars_already_sorted(
    apply: typing.Callable,
):
    """Already sorted data stays sorted."""
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0, 1, 2],
                "ancestor_id": [0, 0, 1],
                "origin_time": [0.0, 1.0, 2.0],
            }
        ),
    )

    result = alifestd_chronological_sort_polars(df_pl)

    assert result["origin_time"].to_list() == [0.0, 1.0, 2.0]


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_chronological_sort_polars_empty(
    apply: typing.Callable,
):
    """Empty dataframe sorting."""
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

    result = alifestd_chronological_sort_polars(df_pl)

    assert result.is_empty()
