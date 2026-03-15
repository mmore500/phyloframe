import os
import typing

import pandas as pd
import polars as pl
import pytest

from phyloframe.legacy import (
    alifestd_mark_num_leaves_sibling_asexual,
    alifestd_to_working_format,
)
from phyloframe.legacy._alifestd_mark_num_leaves_sibling_polars import (
    alifestd_mark_num_leaves_sibling_polars as alifestd_mark_num_leaves_sibling_polars_,
)

from ._impl import enforce_dtype_stability_polars

alifestd_mark_num_leaves_sibling_polars = enforce_dtype_stability_polars(
    alifestd_mark_num_leaves_sibling_polars_,
)

assets_path = os.path.join(os.path.dirname(__file__), "assets")


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_mark_num_leaves_sibling_polars_simple_tree(
    apply: typing.Callable,
):
    """Test a simple tree.

    Tree structure:
        0 (root)
        +-- 1
        |   +-- 3
        |   +-- 4
        +-- 2

    num_leaves: [3, 2, 1, 1, 1]

    num_leaves_sibling:
      0: root => 0
      1: num_leaves[ancestor=0] - num_leaves[1] = 3 - 2 = 1
      2: num_leaves[ancestor=0] - num_leaves[2] = 3 - 1 = 2
      3: num_leaves[ancestor=1] - num_leaves[3] = 2 - 1 = 1
      4: num_leaves[ancestor=1] - num_leaves[4] = 2 - 1 = 1
    """
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0, 1, 2, 3, 4],
                "ancestor_id": [0, 0, 0, 1, 1],
            }
        ),
    )

    result = alifestd_mark_num_leaves_sibling_polars(df_pl).lazy().collect()

    assert result["num_leaves_sibling"].to_list() == [0, 1, 2, 1, 1]


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_mark_num_leaves_sibling_polars_single_node(
    apply: typing.Callable,
):
    """A single root node has 0 sibling leaves."""
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0],
                "ancestor_id": [0],
            }
        ),
    )

    result = alifestd_mark_num_leaves_sibling_polars(df_pl).lazy().collect()

    assert result["num_leaves_sibling"].to_list() == [0]


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_mark_num_leaves_sibling_polars_empty(
    apply: typing.Callable,
):
    """Empty dataframe gets num_leaves_sibling column."""
    df_pl = apply(
        pl.DataFrame(
            {"id": [], "ancestor_id": []},
            schema={"id": pl.Int64, "ancestor_id": pl.Int64},
        ),
    )

    result = alifestd_mark_num_leaves_sibling_polars(df_pl).lazy().collect()

    assert "num_leaves_sibling" in result.columns
    assert result.is_empty()


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_mark_num_leaves_sibling_polars_non_contiguous_ids(
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
    alifestd_mark_num_leaves_sibling_polars(df_pl).lazy().collect()


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_mark_num_leaves_sibling_polars_unsorted(
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
    alifestd_mark_num_leaves_sibling_polars(df_pl).lazy().collect()


@pytest.mark.parametrize(
    "phylogeny_df",
    [
        alifestd_to_working_format(
            pd.read_csv(f"{assets_path}/nk_ecoeaselection.csv")
        ),
        alifestd_to_working_format(
            pd.read_csv(f"{assets_path}/nk_lexicaseselection.csv")
        ),
        alifestd_to_working_format(
            pd.read_csv(f"{assets_path}/nk_tournamentselection.csv")
        ),
    ],
)
@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_mark_num_leaves_sibling_polars_matches_pandas(
    phylogeny_df: pd.DataFrame, apply: typing.Callable
):
    """Verify polars result matches pandas result."""
    result_pd = alifestd_mark_num_leaves_sibling_asexual(
        phylogeny_df,
        mutate=False,
    )

    df_pl = apply(pl.from_pandas(phylogeny_df))
    result_pl = alifestd_mark_num_leaves_sibling_polars(df_pl).lazy().collect()

    assert result_pd["num_leaves_sibling"].tolist() == (
        result_pl["num_leaves_sibling"].to_list()
    )
