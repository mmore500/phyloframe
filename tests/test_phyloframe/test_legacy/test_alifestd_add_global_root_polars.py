import typing

import polars as pl
import pytest

from phyloframe.legacy._alifestd_add_global_root_polars import (
    alifestd_add_global_root_polars as alifestd_add_global_root_polars_,
)

from ._impl import enforce_dtype_stability_polars

alifestd_add_global_root_polars = enforce_dtype_stability_polars(
    alifestd_add_global_root_polars_
)


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_add_global_root_polars_simple_tree(
    apply: typing.Callable,
):
    """Add global root to a simple tree."""
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0, 1, 2],
                "ancestor_id": [0, 0, 0],
            }
        ),
    )

    result = alifestd_add_global_root_polars(df_pl).lazy().collect()

    assert len(result) == 4
    # New root has id=3
    new_root = result.filter(pl.col("id") == 3)
    assert len(new_root) == 1
    assert new_root["ancestor_id"].item() == 3  # self-referencing

    # Old root 0 now points to new root 3
    old_root = result.filter(pl.col("id") == 0)
    assert old_root["ancestor_id"].item() == 3


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_add_global_root_polars_multiple_roots(
    apply: typing.Callable,
):
    """Add global root to a forest with multiple roots."""
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0, 1, 2, 3],
                "ancestor_id": [0, 0, 2, 2],
            }
        ),
    )

    result = alifestd_add_global_root_polars(df_pl).lazy().collect()

    assert len(result) == 5
    # New root has id=4
    new_root = result.filter(pl.col("id") == 4)
    assert new_root["ancestor_id"].item() == 4

    # Both old roots (0 and 2) now point to new root
    assert result.filter(pl.col("id") == 0)["ancestor_id"].item() == 4
    assert result.filter(pl.col("id") == 2)["ancestor_id"].item() == 4


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_add_global_root_polars_empty(
    apply: typing.Callable,
):
    """Empty dataframe gets a single root."""
    df_pl = apply(
        pl.DataFrame(
            {"id": [], "ancestor_id": []},
            schema={"id": pl.Int64, "ancestor_id": pl.Int64},
        ),
    )

    result = alifestd_add_global_root_polars(df_pl).lazy().collect()
    assert len(result) == 1
    assert result["id"].item() == 0
    assert result["ancestor_id"].item() == 0


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_add_global_root_polars_single_node(
    apply: typing.Callable,
):
    """Single node gets a new parent."""
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0],
                "ancestor_id": [0],
            }
        ),
    )

    result = alifestd_add_global_root_polars(df_pl).lazy().collect()
    assert len(result) == 2
