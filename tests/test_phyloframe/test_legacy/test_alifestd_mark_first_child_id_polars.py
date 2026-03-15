import os
import typing

import pandas as pd
import polars as pl
import pytest

from phyloframe.legacy import (
    alifestd_mark_first_child_id_asexual,
    alifestd_to_working_format,
)
from phyloframe.legacy._alifestd_mark_first_child_id_polars import (
    alifestd_mark_first_child_id_polars as alifestd_mark_first_child_id_polars_,
)

from ._impl import enforce_dtype_stability_polars

alifestd_mark_first_child_id_polars = enforce_dtype_stability_polars(
    alifestd_mark_first_child_id_polars_
)

assets_path = os.path.join(os.path.dirname(__file__), "assets")


@pytest.mark.parametrize(
    "phylogeny_df",
    [
        alifestd_to_working_format(
            pd.read_csv(
                f"{assets_path}/example-standard-toy-asexual-phylogeny.csv"
            )
        ),
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
def test_alifestd_mark_first_child_id_polars_fuzz(
    phylogeny_df: pd.DataFrame, apply: typing.Callable
):
    """Verify first_child_id column is correctly added."""
    df_prepared = pl.from_pandas(phylogeny_df)
    df_pl = apply(df_prepared)

    result = alifestd_mark_first_child_id_polars(df_pl).lazy().collect()

    assert "first_child_id" in result.columns
    assert len(result) == len(df_prepared)

    assert result["id"].to_list() == df_prepared["id"].to_list()

    # all first_child_ids should be valid ids
    assert (result["first_child_id"] >= 0).all()
    assert (result["first_child_id"] < len(result)).all()


@pytest.mark.parametrize(
    "phylogeny_df",
    [
        alifestd_to_working_format(
            pd.read_csv(
                f"{assets_path}/example-standard-toy-asexual-phylogeny.csv"
            )
        ),
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
def test_alifestd_mark_first_child_id_polars_matches_pandas(
    phylogeny_df: pd.DataFrame, apply: typing.Callable
):
    """Verify polars result matches pandas result."""
    result_pd = alifestd_mark_first_child_id_asexual(
        phylogeny_df, mutate=False
    )

    df_pl = apply(pl.from_pandas(phylogeny_df))
    result_pl = alifestd_mark_first_child_id_polars(df_pl).lazy().collect()

    assert result_pd["first_child_id"].tolist() == (
        result_pl["first_child_id"].to_list()
    )


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_mark_first_child_id_polars_simple_chain(
    apply: typing.Callable,
):
    """Test a simple chain: 0 -> 1 -> 2."""
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0, 1, 2],
                "ancestor_id": [0, 0, 1],
            }
        ),
    )

    result = alifestd_mark_first_child_id_polars(df_pl).lazy().collect()

    assert result["first_child_id"].to_list() == [1, 2, 2]


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_mark_first_child_id_polars_simple_tree(
    apply: typing.Callable,
):
    """Test a simple tree.

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

    result = alifestd_mark_first_child_id_polars(df_pl).lazy().collect()

    assert result["first_child_id"].to_list() == [1, 3, 2, 3, 4]


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_mark_first_child_id_polars_single_node(
    apply: typing.Callable,
):
    """A single root node's first_child_id is itself."""
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0],
                "ancestor_id": [0],
            }
        ),
    )

    result = alifestd_mark_first_child_id_polars(df_pl).lazy().collect()

    assert result["first_child_id"].to_list() == [0]


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_mark_first_child_id_polars_empty(apply: typing.Callable):
    """Empty dataframe gets first_child_id column."""
    df_pl = apply(
        pl.DataFrame(
            {"id": [], "ancestor_id": []},
            schema={"id": pl.Int64, "ancestor_id": pl.Int64},
        ),
    )

    result = alifestd_mark_first_child_id_polars(df_pl).lazy().collect()

    assert "first_child_id" in result.columns
    assert result.is_empty()


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_mark_first_child_id_polars_non_contiguous_ids(
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
    alifestd_mark_first_child_id_polars(df_pl).lazy().collect()


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_mark_first_child_id_polars_unsorted(
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
    alifestd_mark_first_child_id_polars(df_pl).lazy().collect()


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_mark_first_child_id_polars_depth_first(
    apply: typing.Callable,
):
    """Navigate the tree depth-first using first_child_id.

    Tree:
        0 (root)
        +-- 1
        |   +-- 3
        |   |   +-- 7
        |   |   +-- 8
        |   +-- 4
        +-- 2
        |   +-- 5
        |   +-- 6
    """
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0, 1, 2, 3, 4, 5, 6, 7, 8],
                "ancestor_id": [0, 0, 0, 1, 1, 2, 2, 3, 3],
            }
        ),
    )

    result = alifestd_mark_first_child_id_polars(df_pl).lazy().collect()
    first_child = result["first_child_id"].to_list()

    # navigate leftmost path: 0 -> 1 -> 3 -> 7 (leaf)
    cur = 0
    path = [cur]
    while first_child[cur] != cur:
        cur = first_child[cur]
        path.append(cur)

    assert path == [0, 1, 3, 7]
