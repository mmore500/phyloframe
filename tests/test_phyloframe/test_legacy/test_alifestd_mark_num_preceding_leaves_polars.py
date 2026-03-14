import typing

import polars as pl
import pytest

from phyloframe.legacy._alifestd_mark_num_preceding_leaves_polars import (
    alifestd_mark_num_preceding_leaves_polars as alifestd_mark_num_preceding_leaves_polars_,
)

from ._impl import enforce_dtype_stability_polars

alifestd_mark_num_preceding_leaves_polars = enforce_dtype_stability_polars(
    alifestd_mark_num_preceding_leaves_polars_
)


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_mark_num_preceding_leaves_polars_simple_tree(
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

    result = alifestd_mark_num_preceding_leaves_polars(df_pl).lazy().collect()

    assert "num_preceding_leaves" in result.columns
    assert len(result) == 5


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_mark_num_preceding_leaves_polars_single_node(
    apply: typing.Callable,
):
    """A single root node."""
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0],
                "ancestor_id": [0],
            }
        ),
    )

    result = alifestd_mark_num_preceding_leaves_polars(df_pl).lazy().collect()

    assert result["num_preceding_leaves"].to_list() == [0]


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_mark_num_preceding_leaves_polars_empty(
    apply: typing.Callable,
):
    """Empty dataframe gets num_preceding_leaves column."""
    df_pl = apply(
        pl.DataFrame(
            {"id": [], "ancestor_id": []},
            schema={"id": pl.Int64, "ancestor_id": pl.Int64},
        ),
    )

    result = alifestd_mark_num_preceding_leaves_polars(df_pl).lazy().collect()

    assert "num_preceding_leaves" in result.columns
    assert result.is_empty()


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_mark_num_preceding_leaves_polars_non_contiguous_ids(
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
    with pytest.raises(NotImplementedError):
        alifestd_mark_num_preceding_leaves_polars(df_pl).lazy().collect()


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_mark_num_preceding_leaves_polars_unsorted(
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
    with pytest.raises(NotImplementedError):
        alifestd_mark_num_preceding_leaves_polars(df_pl).lazy().collect()
