import typing

import polars as pl
import pytest

from phyloframe.legacy._alifestd_join_roots_polars import (
    alifestd_join_roots_polars as alifestd_join_roots_polars_,
)

from ._impl import enforce_dtype_stability_polars

alifestd_join_roots_polars = enforce_dtype_stability_polars(
    alifestd_join_roots_polars_
)


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_join_roots_polars_single_root(
    apply: typing.Callable,
):
    """Single root tree stays the same."""
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0, 1, 2],
                "ancestor_id": [0, 0, 0],
            }
        ),
    )

    result = alifestd_join_roots_polars(df_pl).lazy().collect()

    assert len(result) == 3
    assert result["is_root"].to_list().count(True) == 1


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_join_roots_polars_multiple_roots(
    apply: typing.Callable,
):
    """Multiple roots get joined to one."""
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0, 1, 2, 3],
                "ancestor_id": [0, 0, 2, 2],
            }
        ),
    )

    result = alifestd_join_roots_polars(df_pl).lazy().collect()

    # Only one root remains
    assert result["is_root"].to_list().count(True) == 1
    # The oldest root (lowest id) is root 0
    root_row = result.filter(pl.col("is_root"))
    assert root_row["id"].item() == 0


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_join_roots_polars_with_origin_time(
    apply: typing.Callable,
):
    """With origin_time, oldest root is by lowest origin_time."""
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0, 1, 2, 3],
                "ancestor_id": [0, 0, 2, 2],
                "origin_time": [5.0, 1.0, 1.0, 2.0],
            }
        ),
    )

    result = alifestd_join_roots_polars(df_pl).lazy().collect()

    # Root 2 has origin_time=1.0, same as non-root node 1
    # Root 0 has origin_time=5.0
    # So root 2 should be the oldest root (lowest origin_time)
    root_row = result.filter(pl.col("is_root"))
    assert root_row["id"].item() == 2


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_join_roots_polars_single_node(
    apply: typing.Callable,
):
    """Single node stays unchanged."""
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0],
                "ancestor_id": [0],
            }
        ),
    )

    result = alifestd_join_roots_polars(df_pl).lazy().collect()
    assert len(result) == 1


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_join_roots_polars_empty(
    apply: typing.Callable,
):
    """Empty dataframe is handled."""
    df_pl = apply(
        pl.DataFrame(
            {"id": [], "ancestor_id": []},
            schema={"id": pl.Int64, "ancestor_id": pl.Int64},
        ),
    )

    result = alifestd_join_roots_polars(df_pl).lazy().collect()
    assert result.is_empty()
