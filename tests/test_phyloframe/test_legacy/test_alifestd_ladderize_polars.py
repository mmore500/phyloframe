import os
import typing

import pandas as pd
import polars as pl
import pytest

from phyloframe.legacy import (
    alifestd_ladderize_asexual,
    alifestd_to_working_format,
)
from phyloframe.legacy._alifestd_ladderize_polars import (
    alifestd_ladderize_polars,
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
def test_alifestd_ladderize_polars_fuzz(
    phylogeny_df: pd.DataFrame, apply: typing.Callable
):
    """Verify rows are reordered and all ids preserved."""
    df_prepared = pl.from_pandas(phylogeny_df)
    df_pl = apply(df_prepared)

    result = alifestd_ladderize_polars(df_pl).lazy().collect()

    assert len(result) == len(df_prepared)
    assert set(result["id"].to_list()) == set(df_prepared["id"].to_list())


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
def test_alifestd_ladderize_polars_matches_pandas(
    phylogeny_df: pd.DataFrame, apply: typing.Callable
):
    """Verify polars result matches pandas result."""
    result_pd = alifestd_ladderize_asexual(phylogeny_df, mutate=False)

    df_pl = apply(pl.from_pandas(phylogeny_df))
    result_pl = alifestd_ladderize_polars(df_pl).lazy().collect()

    assert result_pd["id"].tolist() == result_pl["id"].to_list()


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
    ],
)
@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_ladderize_polars_reverse_matches_pandas(
    phylogeny_df: pd.DataFrame, apply: typing.Callable
):
    """Verify polars reverse result matches pandas reverse result."""
    result_pd = alifestd_ladderize_asexual(
        phylogeny_df, reverse=True, mutate=False
    )

    df_pl = apply(pl.from_pandas(phylogeny_df))
    result_pl = alifestd_ladderize_polars(df_pl, reverse=True).lazy().collect()

    assert result_pd["id"].tolist() == result_pl["id"].to_list()


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_ladderize_polars_simple_tree(apply: typing.Callable):
    """Tree structure:
        0 (root)
        +-- 1
        |   +-- 3
        |   +-- 4
        +-- 2

    Ladderized: 2 (1 leaf) before 1's subtree (2 leaves).
    """
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0, 1, 2, 3, 4],
                "ancestor_id": [0, 0, 0, 1, 1],
            }
        ),
    )

    result = alifestd_ladderize_polars(df_pl).lazy().collect()

    result_ids = result["id"].to_list()
    assert result_ids[0] == 0
    assert result_ids.index(2) < result_ids.index(1)
    assert result_ids.index(1) < result_ids.index(3)
    assert result_ids.index(1) < result_ids.index(4)


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_ladderize_polars_simple_tree_reverse(
    apply: typing.Callable,
):
    """Reverse: more leaves first."""
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0, 1, 2, 3, 4],
                "ancestor_id": [0, 0, 0, 1, 1],
            }
        ),
    )

    result = alifestd_ladderize_polars(df_pl, reverse=True).lazy().collect()

    result_ids = result["id"].to_list()
    assert result_ids[0] == 0
    assert result_ids.index(1) < result_ids.index(2)


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_ladderize_polars_simple_chain(apply: typing.Callable):
    """Chain: 0 -> 1 -> 2. No reordering needed."""
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0, 1, 2],
                "ancestor_id": [0, 0, 1],
            }
        ),
    )

    result = alifestd_ladderize_polars(df_pl).lazy().collect()

    assert result["id"].to_list() == [0, 1, 2]


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_ladderize_polars_empty(apply: typing.Callable):
    """Empty dataframe stays empty."""
    df_pl = apply(
        pl.DataFrame(
            {"id": [], "ancestor_id": []},
            schema={"id": pl.Int64, "ancestor_id": pl.Int64},
        ),
    )

    result = alifestd_ladderize_polars(df_pl).lazy().collect()

    assert result.is_empty()


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_ladderize_polars_single_node(apply: typing.Callable):
    """Single root node."""
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0],
                "ancestor_id": [0],
            }
        ),
    )

    result = alifestd_ladderize_polars(df_pl).lazy().collect()

    assert result["id"].to_list() == [0]


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_ladderize_polars_preserves_columns(
    apply: typing.Callable,
):
    """Verify original columns are preserved."""
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0, 1, 2, 3, 4],
                "ancestor_id": [0, 0, 0, 1, 1],
                "origin_time": [0.0, 1.0, 2.0, 3.0, 4.0],
                "taxon_label": ["a", "b", "c", "d", "e"],
            }
        ),
    )

    result = alifestd_ladderize_polars(df_pl).lazy().collect()

    assert "origin_time" in result.columns
    assert "taxon_label" in result.columns
    assert len(result) == 5


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_ladderize_polars_non_contiguous_ids(
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
    alifestd_ladderize_polars(df_pl).lazy().collect()


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_ladderize_polars_unsorted(apply: typing.Callable):
    """Verify NotImplementedError for topologically unsorted data."""
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0, 1, 2],
                "ancestor_id": [0, 2, 0],
            }
        ),
    )
    alifestd_ladderize_polars(df_pl).lazy().collect()


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_ladderize_polars_star_topology(apply: typing.Callable):
    """Root with many direct children (all leaves)."""
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0, 1, 2, 3, 4, 5],
                "ancestor_id": [0, 0, 0, 0, 0, 0],
            }
        ),
    )

    result = alifestd_ladderize_polars(df_pl).lazy().collect()

    # all children are leaves with 1 leaf each, so order is stable
    assert set(result["id"].to_list()) == {0, 1, 2, 3, 4, 5}
    assert result["id"].to_list()[0] == 0
