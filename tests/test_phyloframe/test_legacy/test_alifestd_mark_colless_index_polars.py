import typing

import polars as pl
import pytest

from phyloframe.legacy._alifestd_mark_colless_index_polars import (
    alifestd_mark_colless_index_polars as alifestd_mark_colless_index_polars_,
)

from ._impl import enforce_dtype_stability_polars

alifestd_mark_colless_index_polars = enforce_dtype_stability_polars(
    alifestd_mark_colless_index_polars_
)


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_mark_colless_index_polars_simple_tree(
    apply: typing.Callable,
):
    """Test a simple bifurcating tree.

    Tree structure:
        0 (root)
        +-- 1
        |   +-- 3
        |   +-- 4
        +-- 2
    """
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0, 1, 2, 3, 4],
                "ancestor_id": [0, 0, 0, 1, 1],
            }
        ),
    )

    result = alifestd_mark_colless_index_polars(df_pl).lazy().collect()

    assert "colless_index" in result.columns
    assert len(result) == 5
    # Root has 3 leaves vs subtrees: left=2 leaves, right=1 leaf => |2-1|=1
    # Node 1 has 2 leaves, balanced => 0
    # Leaves have 0
    assert result["colless_index"].to_list() == [1, 0, 0, 0, 0]


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_mark_colless_index_polars_single_node(
    apply: typing.Callable,
):
    """A single root node has colless_index of 0."""
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0],
                "ancestor_id": [0],
            }
        ),
    )

    result = alifestd_mark_colless_index_polars(df_pl).lazy().collect()

    assert result["colless_index"].to_list() == [0]


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_mark_colless_index_polars_empty(apply: typing.Callable):
    """Empty dataframe gets colless_index column."""
    df_pl = apply(
        pl.DataFrame(
            {"id": [], "ancestor_id": []},
            schema={"id": pl.Int64, "ancestor_id": pl.Int64},
        ),
    )

    result = alifestd_mark_colless_index_polars(df_pl).lazy().collect()

    assert "colless_index" in result.columns
    assert result.is_empty()


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_mark_colless_index_polars_non_contiguous_ids(
    apply: typing.Callable,
):
    """Verify NotImplementedError for non-contiguous ids."""
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0, 2, 5],
                "ancestor_id": [0, 0, 2],
            }
        ),
    )
    alifestd_mark_colless_index_polars(df_pl).lazy().collect()


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_mark_colless_index_polars_unsorted(
    apply: typing.Callable,
):
    """Verify NotImplementedError for topologically unsorted data."""
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0, 1, 2],
                "ancestor_id": [0, 2, 0],
            }
        ),
    )
    alifestd_mark_colless_index_polars(df_pl).lazy().collect()
