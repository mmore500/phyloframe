import typing

import polars as pl
import pytest

from phyloframe.legacy._alifestd_is_chronologically_ordered_polars import (
    alifestd_is_chronologically_ordered_polars,
)


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_is_chronologically_ordered_polars_true(
    apply: typing.Callable,
):
    """Properly ordered tree returns True."""
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0, 1, 2],
                "ancestor_id": [0, 0, 1],
                "origin_time": [0.0, 1.0, 2.0],
            }
        ),
    )

    assert alifestd_is_chronologically_ordered_polars(df_pl) is True


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_is_chronologically_ordered_polars_false(
    apply: typing.Callable,
):
    """Tree with ancestor after child returns False."""
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0, 1, 2],
                "ancestor_id": [0, 0, 1],
                "origin_time": [5.0, 1.0, 0.0],
            }
        ),
    )

    assert alifestd_is_chronologically_ordered_polars(df_pl) is False


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_is_chronologically_ordered_polars_empty(
    apply: typing.Callable,
):
    """Empty dataframe is chronologically ordered."""
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

    assert alifestd_is_chronologically_ordered_polars(df_pl) is True
