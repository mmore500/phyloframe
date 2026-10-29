import os
import typing

import pandas as pd
import polars as pl
import pytest

from phyloframe.legacy import (
    alifestd_collapse_unifurcations_polars,
    alifestd_mark_unifurcation_length_asexual,
    alifestd_mark_unifurcation_length_polars,
    alifestd_to_working_format,
)

assets_path = os.path.join(os.path.dirname(__file__), "assets")


def _make_df() -> pl.DataFrame:
    """Tree:
    0 (root)
    +-- 1
        +-- 2
            +-- 3
            |   +-- 5
            |       +-- 6
            +-- 4
    """
    return pl.DataFrame(
        {
            "id": [0, 1, 2, 3, 4, 5, 6],
            "ancestor_id": [0, 0, 1, 2, 2, 3, 5],
            "w": [0, 4, 1, 0, 2, 0, 3],
        },
    )


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_simple_tree(apply: typing.Callable):
    res = (
        alifestd_mark_unifurcation_length_polars(apply(_make_df()))
        .lazy()
        .collect()
    )
    assert res["unifurcation_length"].to_list() == [0, 0, 1, 0, 0, 1, 2]


def test_weights():
    res = alifestd_mark_unifurcation_length_polars(_make_df(), weights="w")
    assert res["unifurcation_length"].to_list() == [0, 0, 4, 0, 0, 0, 0]


def test_weights_expr_accumulate_collapsed():
    # accumulate counts across collapses: own count plus, for each
    # bypassed unifurcation, its own count and itself
    df = alifestd_mark_unifurcation_length_polars(
        _make_df(), weights=pl.col("w") + 1
    )
    df = df.with_columns(pl.col("w") + pl.col("unifurcation_length"))
    res = alifestd_collapse_unifurcations_polars(df)
    assert res["id"].to_list() == [0, 2, 4, 6]
    assert res["w"].to_list() == [0, 6, 2, 5]


def test_mark_as():
    res = alifestd_mark_unifurcation_length_polars(
        _make_df(), mark_as="blueberry"
    )
    assert "blueberry" in res.columns
    assert "unifurcation_length" not in res.columns


def test_unifurcating_root_not_counted():
    df = pl.DataFrame({"id": [0, 1, 2], "ancestor_id": [0, 0, 1]})
    res = alifestd_mark_unifurcation_length_polars(df)
    assert res["unifurcation_length"].to_list() == [0, 0, 1]


def test_multiple_roots():
    df = pl.DataFrame(
        {"id": [0, 1, 2, 3, 4], "ancestor_id": [0, 1, 0, 2, 1]},
    )
    res = alifestd_mark_unifurcation_length_polars(df)
    assert res["unifurcation_length"].to_list() == [0, 0, 0, 1, 0]


def test_empty():
    df = pl.DataFrame(
        {"id": [], "ancestor_id": []},
        schema={"id": pl.Int64, "ancestor_id": pl.Int64},
    )
    res = alifestd_mark_unifurcation_length_polars(df)
    assert "unifurcation_length" in res.columns
    assert len(res) == 0


def test_missing_weights_column():
    with pytest.raises(ValueError):
        alifestd_mark_unifurcation_length_polars(_make_df(), weights="nope")


def test_non_contiguous_ids():
    df = pl.DataFrame({"id": [0, 2], "ancestor_id": [0, 0]})
    with pytest.raises(NotImplementedError):
        alifestd_mark_unifurcation_length_polars(df)


@pytest.mark.parametrize(
    "phylogeny_df",
    [
        pd.read_csv(
            f"{assets_path}/example-standard-toy-asexual-phylogeny.csv"
        ),
        pd.read_csv(f"{assets_path}/nk_tournamentselection.csv"),
    ],
)
def test_matches_pandas_and_collapse(phylogeny_df: pd.DataFrame):
    phylogeny_df = alifestd_to_working_format(phylogeny_df)
    phylogeny_df = phylogeny_df.drop(columns=["ancestor_list"])
    expected = alifestd_mark_unifurcation_length_asexual(phylogeny_df)
    df_pl = pl.from_pandas(phylogeny_df)
    res = alifestd_mark_unifurcation_length_polars(df_pl)
    assert (
        res["unifurcation_length"].to_list()
        == expected["unifurcation_length"].tolist()
    )

    collapsed = alifestd_collapse_unifurcations_polars(res)
    assert collapsed["unifurcation_length"].sum() == len(df_pl) - len(
        collapsed
    )
