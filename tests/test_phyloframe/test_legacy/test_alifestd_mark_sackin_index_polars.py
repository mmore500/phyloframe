import os
import typing

import pandas as pd
import polars as pl
import pytest

from phyloframe.legacy import (
    alifestd_mark_sackin_index_asexual,
    alifestd_to_working_format,
)
from phyloframe.legacy._alifestd_mark_sackin_index_polars import (
    alifestd_mark_sackin_index_polars as alifestd_mark_sackin_index_polars_,
)

from ._impl import enforce_dtype_stability_polars

alifestd_mark_sackin_index_polars = enforce_dtype_stability_polars(
    alifestd_mark_sackin_index_polars_
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
def test_alifestd_mark_sackin_index_polars_matches_pandas(
    phylogeny_df: pd.DataFrame, apply: typing.Callable
):
    """Verify polars result matches pandas result."""
    result_pd = alifestd_mark_sackin_index_asexual(phylogeny_df, mutate=False)

    df_pl = apply(pl.from_pandas(phylogeny_df))
    result_pl = alifestd_mark_sackin_index_polars(df_pl).lazy().collect()

    assert result_pd["sackin_index"].tolist() == (
        result_pl["sackin_index"].to_list()
    )


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_mark_sackin_index_polars_simple_chain(
    apply: typing.Callable,
):
    """Test a simple chain: 0 -> 1 -> 2.

    Depths: 0=0, 1=1, 2=2
    Leaf is node 2 (depth 2).
    sackin[2] = 0 (leaf)
    sackin[1] = sackin[2] + num_leaves[2] = 0 + 1 = 1
    sackin[0] = sackin[1] + num_leaves[1] = 1 + 2 = 3
    Wait, that's not right. num_leaves[1] = 1 (just leaf 2).
    sackin[0] = sackin[1] + num_leaves[1] = 1 + 1 = 2
    """
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0, 1, 2],
                "ancestor_id": [0, 0, 1],
            }
        ),
    )

    result = alifestd_mark_sackin_index_polars(df_pl).lazy().collect()

    assert result["sackin_index"].to_list() == [2, 1, 0]


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_mark_sackin_index_polars_simple_tree(
    apply: typing.Callable,
):
    """Test a balanced tree.

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

    result = alifestd_mark_sackin_index_polars(df_pl).lazy().collect()

    # leaves: 2, 3, 4 all have sackin_index 0
    assert result["sackin_index"][2] == 0
    assert result["sackin_index"][3] == 0
    assert result["sackin_index"][4] == 0
    # sackin[1] = sackin[3] + num_leaves[3] + sackin[4] + num_leaves[4]
    #           = 0 + 1 + 0 + 1 = 2
    assert result["sackin_index"][1] == 2
    # sackin[0] = sackin[1] + num_leaves[1] + sackin[2] + num_leaves[2]
    #           = 2 + 2 + 0 + 1 = 5
    assert result["sackin_index"][0] == 5


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_mark_sackin_index_polars_single_node(
    apply: typing.Callable,
):
    """A single root has sackin_index 0."""
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0],
                "ancestor_id": [0],
            }
        ),
    )

    result = alifestd_mark_sackin_index_polars(df_pl).lazy().collect()

    assert result["sackin_index"].to_list() == [0]


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_mark_sackin_index_polars_empty(apply: typing.Callable):
    """Empty dataframe gets sackin_index column."""
    df_pl = apply(
        pl.DataFrame(
            {"id": [], "ancestor_id": []},
            schema={"id": pl.Int64, "ancestor_id": pl.Int64},
        ),
    )

    result = alifestd_mark_sackin_index_polars(df_pl).lazy().collect()

    assert "sackin_index" in result.columns
    assert result.is_empty()


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_mark_sackin_index_polars_non_contiguous_ids(
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
        alifestd_mark_sackin_index_polars(df_pl).lazy().collect()


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_mark_sackin_index_polars_unsorted(
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
        alifestd_mark_sackin_index_polars(df_pl).lazy().collect()
