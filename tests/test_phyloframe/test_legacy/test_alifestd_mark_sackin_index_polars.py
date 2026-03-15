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
    alifestd_mark_sackin_index_polars_,
)

assets_path = os.path.join(os.path.dirname(__file__), "assets")


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
def test_matches_pandas(phylogeny_df: pd.DataFrame, apply: typing.Callable):
    result_pd = alifestd_mark_sackin_index_asexual(phylogeny_df, mutate=False)
    df_pl = apply(pl.from_pandas(phylogeny_df))
    result_pl = alifestd_mark_sackin_index_polars(df_pl).lazy().collect()
    assert (
        result_pd["sackin_index"].tolist()
        == result_pl["sackin_index"].to_list()
    )


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_simple_tree(apply: typing.Callable):
    """Tree: 0(root) -> 1 -> 3, 0 -> 1 -> 4, 0 -> 2"""
    df = apply(
        pl.DataFrame({"id": [0, 1, 2, 3, 4], "ancestor_id": [0, 0, 0, 1, 1]})
    )
    result = alifestd_mark_sackin_index_polars(df).lazy().collect()
    assert "sackin_index" in result.columns
    # Leaves (2,3,4): sackin=0
    # Node 1: sackin = (0+1) + (0+1) = 2  (two leaf children)
    # Node 0: sackin = (2+2) + (0+1) = 5  (node 1 has 2 leaves, node 2 has 1 leaf)
    assert result["sackin_index"].to_list() == [5, 2, 0, 0, 0]


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_empty(apply: typing.Callable):
    df = apply(
        pl.DataFrame(
            {"id": [], "ancestor_id": []},
            schema={"id": pl.Int64, "ancestor_id": pl.Int64},
        )
    )
    result = alifestd_mark_sackin_index_polars(df).lazy().collect()
    assert "sackin_index" in result.columns
    assert result.is_empty()


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_non_contiguous_ids(apply: typing.Callable):
    df = apply(pl.DataFrame({"id": [0, 2, 5], "ancestor_id": [0, 0, 2]}))
    alifestd_mark_sackin_index_polars(df).lazy().collect()
