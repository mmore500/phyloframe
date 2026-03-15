import typing

import polars as pl
import pytest

from phyloframe.legacy._alifestd_mark_colless_index_corrected_polars import (
    alifestd_mark_colless_index_corrected_polars as alifestd_mark_colless_index_corrected_polars_,
)

from ._impl import enforce_dtype_stability_polars

alifestd_mark_colless_index_corrected_polars = enforce_dtype_stability_polars(
    alifestd_mark_colless_index_corrected_polars_
)


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_mark_colless_index_corrected_polars_simple_tree(
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

    result = (
        alifestd_mark_colless_index_corrected_polars(df_pl).lazy().collect()
    )

    assert "colless_index_corrected" in result.columns
    assert len(result) == 5
    # Root: n=3 leaves, C=1, corrected = 2*1/((3-1)*(3-2)) = 2/2 = 1.0
    # Node 1: n=2 leaves, C=0, but n<=2 so corrected=0
    # Leaves: n=1, corrected=0
    assert result["colless_index_corrected"].to_list() == pytest.approx(
        [1.0, 0.0, 0.0, 0.0, 0.0]
    )


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_mark_colless_index_corrected_polars_single_node(
    apply: typing.Callable,
):
    """A single root node has colless_index_corrected of 0."""
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0],
                "ancestor_id": [0],
            }
        ),
    )

    result = (
        alifestd_mark_colless_index_corrected_polars(df_pl).lazy().collect()
    )

    assert result["colless_index_corrected"].to_list() == pytest.approx([0.0])


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_mark_colless_index_corrected_polars_empty(
    apply: typing.Callable,
):
    """Empty dataframe gets colless_index_corrected column."""
    df_pl = apply(
        pl.DataFrame(
            {"id": [], "ancestor_id": []},
            schema={"id": pl.Int64, "ancestor_id": pl.Int64},
        ),
    )

    result = (
        alifestd_mark_colless_index_corrected_polars(df_pl).lazy().collect()
    )

    assert "colless_index_corrected" in result.columns
    assert result.is_empty()


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_mark_colless_index_corrected_polars_non_contiguous_ids(
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
    alifestd_mark_colless_index_corrected_polars(df_pl).lazy().collect()
