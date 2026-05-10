import os
import typing

import pandas as pd
import polars as pl
import pytest

from phyloframe.legacy import (
    alifestd_aggregate_phylogenies,
    alifestd_collapse_unifurcations,
    alifestd_make_balanced_bifurcating,
    alifestd_make_comb,
    alifestd_make_empty_polars,
    alifestd_make_leaf_split,
    alifestd_test_leaves_isomorphic_asexual,
)
from phyloframe.legacy._alifestd_test_leaves_isomorphic_polars import (
    alifestd_test_leaves_isomorphic_polars,
)

assets_path = os.path.join(os.path.dirname(__file__), "assets")


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_empty(apply: typing.Callable):
    mt = alifestd_make_empty_polars().with_columns(
        taxon_label=pl.lit(None, dtype=pl.Int64),
    )
    assert alifestd_test_leaves_isomorphic_polars(
        apply(mt), apply(mt), "taxon_label"
    )


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_does_not_mutate(apply: typing.Callable):
    original_df = pl.from_pandas(
        pd.read_csv(f"{assets_path}/nk_ecoeaselection.csv")
    )
    original_df = original_df.with_columns(taxon_label=pl.col("id"))
    df = original_df.clone()
    alifestd_test_leaves_isomorphic_polars(apply(df), apply(df), "taxon_label")
    assert df.equals(original_df)


@pytest.mark.parametrize(
    "phylogeny_df",
    [
        pd.read_csv(
            f"{assets_path}/example-standard-toy-asexual-phylogeny.csv"
        ),
        pd.read_csv(f"{assets_path}/nk_ecoeaselection.csv"),
        pd.read_csv(f"{assets_path}/nk_lexicaseselection.csv"),
        alifestd_aggregate_phylogenies(
            [
                pd.read_csv(f"{assets_path}/nk_ecoeaselection.csv"),
                pd.read_csv(f"{assets_path}/nk_lexicaseselection.csv"),
            ]
        ),
        pd.read_csv(f"{assets_path}/nk_tournamentselection.csv"),
    ],
)
def test_fuzz_positive_int_labels(phylogeny_df: pd.DataFrame):
    phylogeny_df = phylogeny_df.copy()
    phylogeny_df["taxon_label"] = phylogeny_df["id"]
    df_pl = pl.from_pandas(phylogeny_df)

    assert alifestd_test_leaves_isomorphic_polars(df_pl, df_pl, "taxon_label")
    assert alifestd_test_leaves_isomorphic_polars(
        df_pl,
        pl.from_pandas(alifestd_collapse_unifurcations(phylogeny_df)),
        "taxon_label",
    )
    assert alifestd_test_leaves_isomorphic_polars(
        df_pl,
        df_pl.sample(fraction=1.0, shuffle=True, seed=1),
        "taxon_label",
    )


@pytest.mark.parametrize(
    "phylogeny_df",
    [
        pd.read_csv(
            f"{assets_path}/example-standard-toy-asexual-phylogeny.csv"
        ),
        pd.read_csv(f"{assets_path}/nk_ecoeaselection.csv"),
        pd.read_csv(f"{assets_path}/nk_lexicaseselection.csv"),
        alifestd_aggregate_phylogenies(
            [
                pd.read_csv(f"{assets_path}/nk_ecoeaselection.csv"),
                pd.read_csv(f"{assets_path}/nk_lexicaseselection.csv"),
            ]
        ),
        pd.read_csv(f"{assets_path}/nk_tournamentselection.csv"),
    ],
)
def test_fuzz_positive_str_labels(phylogeny_df: pd.DataFrame):
    phylogeny_df = phylogeny_df.copy()
    phylogeny_df["taxon_label"] = phylogeny_df["id"].astype(str)
    df_pl = pl.from_pandas(phylogeny_df)

    assert alifestd_test_leaves_isomorphic_polars(df_pl, df_pl, "taxon_label")
    assert alifestd_test_leaves_isomorphic_polars(
        df_pl,
        pl.from_pandas(alifestd_collapse_unifurcations(phylogeny_df)),
        "taxon_label",
    )
    assert alifestd_test_leaves_isomorphic_polars(
        df_pl,
        df_pl.sample(fraction=1.0, shuffle=True, seed=2),
        "taxon_label",
    )


def test_negative_different_topologies():
    df1 = pd.read_csv(f"{assets_path}/nk_lexicaseselection.csv")
    df2 = pd.read_csv(f"{assets_path}/nk_ecoeaselection.csv")
    df1["taxon_label"] = df1["id"]
    df2["taxon_label"] = df2["id"]
    assert not alifestd_test_leaves_isomorphic_polars(
        pl.from_pandas(df1), pl.from_pandas(df2), "taxon_label"
    )


def test_negative_tweaked():
    df1 = pd.read_csv(f"{assets_path}/nk_ecoeaselection.csv")
    df2 = pd.read_csv(f"{assets_path}/nk_ecoeaselection_tweaked.csv")
    df1["taxon_label"] = df1["id"].astype(str)
    df2["taxon_label"] = df2["id"].astype(str)
    assert not alifestd_test_leaves_isomorphic_polars(
        pl.from_pandas(df1), pl.from_pandas(df2), "taxon_label"
    )


@pytest.mark.parametrize(
    "phylogeny_df",
    [
        pd.read_csv(
            f"{assets_path}/example-standard-toy-asexual-phylogeny.csv"
        ),
        pd.read_csv(f"{assets_path}/nk_ecoeaselection.csv"),
        pd.read_csv(f"{assets_path}/nk_lexicaseselection.csv"),
        pd.read_csv(f"{assets_path}/nk_tournamentselection.csv"),
        alifestd_make_balanced_bifurcating(depth=4),
        alifestd_make_balanced_bifurcating(depth=5),
        alifestd_make_comb(n_leaves=8),
        alifestd_make_leaf_split(n_leaves=16, seed=0),
        alifestd_make_leaf_split(n_leaves=32, seed=42),
        alifestd_make_leaf_split(n_leaves=64, seed=7),
    ],
)
def test_fuzz_against_asexual_implementation(phylogeny_df: pd.DataFrame):
    """Fuzz: polars result should agree with pandas asexual implementation.

    Compares results across self-comparisons, shuffles, unifurcation
    collapse, and cross-tree comparisons.
    """
    phylogeny_df = phylogeny_df.copy()
    phylogeny_df["taxon_label"] = phylogeny_df["id"].astype(str)
    df_pl = pl.from_pandas(phylogeny_df)

    # self-comparison
    assert alifestd_test_leaves_isomorphic_polars(
        df_pl, df_pl, "taxon_label"
    ) == alifestd_test_leaves_isomorphic_asexual(
        phylogeny_df, phylogeny_df, "taxon_label"
    )

    # collapsed-vs-original
    collapsed_pd = alifestd_collapse_unifurcations(phylogeny_df)
    collapsed_pd["taxon_label"] = collapsed_pd["id"].astype(str)
    collapsed_pl = pl.from_pandas(collapsed_pd)
    assert alifestd_test_leaves_isomorphic_polars(
        df_pl, collapsed_pl, "taxon_label"
    ) == alifestd_test_leaves_isomorphic_asexual(
        phylogeny_df, collapsed_pd, "taxon_label"
    )

    # shuffled
    shuffled_pl = df_pl.sample(fraction=1.0, shuffle=True, seed=3)
    shuffled_pd = shuffled_pl.to_pandas()
    assert alifestd_test_leaves_isomorphic_polars(
        df_pl, shuffled_pl, "taxon_label"
    ) == alifestd_test_leaves_isomorphic_asexual(
        phylogeny_df, shuffled_pd, "taxon_label"
    )


@pytest.mark.parametrize(
    "seed",
    [0, 1, 2, 7, 42, 100, 314, 2718],
)
def test_fuzz_random_trees_positive(seed: int):
    """Two trees built from the same Yule sample should be isomorphic."""
    df_a = alifestd_make_leaf_split(n_leaves=24, seed=seed)
    df_a["taxon_label"] = df_a["id"].astype(str)
    df_b = df_a.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    df_a_pl = pl.from_pandas(df_a)
    df_b_pl = pl.from_pandas(df_b)
    assert alifestd_test_leaves_isomorphic_polars(
        df_a_pl, df_b_pl, "taxon_label"
    )
    assert alifestd_test_leaves_isomorphic_asexual(df_a, df_b, "taxon_label")


@pytest.mark.parametrize(
    "seed_pair",
    [(0, 1), (2, 3), (4, 5), (7, 42), (100, 314)],
)
def test_fuzz_random_trees_negative(seed_pair: tuple):
    """Two independently-sampled Yule trees of the same leaf set are very
    unlikely to be isomorphic. We use an even number of leaves >= 6, which
    bounds the probability of accidental isomorphism well below 1.
    """
    s1, s2 = seed_pair
    df_a = alifestd_make_leaf_split(n_leaves=20, seed=s1)
    df_b = alifestd_make_leaf_split(n_leaves=20, seed=s2)

    leaves_a = set(df_a["id"]) - set(
        df_a["ancestor_list"].str.extract(r"\[(\d+)\]").dropna()[0].astype(int)
    )
    leaves_b = set(df_b["id"]) - set(
        df_b["ancestor_list"].str.extract(r"\[(\d+)\]").dropna()[0].astype(int)
    )
    # the two yule samples should have the same number of leaves
    assert len(leaves_a) == len(leaves_b) == 20

    # use the same taxon label set so the test focuses on topology
    df_a = df_a.copy()
    df_b = df_b.copy()
    df_a["taxon_label"] = "x"
    df_b["taxon_label"] = "x"
    for new, old in zip(sorted(leaves_a), range(20)):
        df_a.loc[df_a["id"] == new, "taxon_label"] = f"leaf_{old}"
    for new, old in zip(sorted(leaves_b), range(20)):
        df_b.loc[df_b["id"] == new, "taxon_label"] = f"leaf_{old}"

    pl_a = pl.from_pandas(df_a)
    pl_b = pl.from_pandas(df_b)

    polars_result = alifestd_test_leaves_isomorphic_polars(
        pl_a, pl_b, "taxon_label"
    )
    pandas_result = alifestd_test_leaves_isomorphic_asexual(
        df_a, df_b, "taxon_label"
    )
    assert polars_result == pandas_result


def test_negative_disjoint_leaves():
    """When leaf taxon labels differ the test should return False."""
    df_a = alifestd_make_leaf_split(n_leaves=8, seed=0)
    df_a["taxon_label"] = "a-" + df_a["id"].astype(str)
    df_b = alifestd_make_leaf_split(n_leaves=8, seed=0)
    df_b["taxon_label"] = "b-" + df_b["id"].astype(str)
    pl_a = pl.from_pandas(df_a)
    pl_b = pl.from_pandas(df_b)
    assert not alifestd_test_leaves_isomorphic_polars(
        pl_a, pl_b, "taxon_label"
    )
    assert not alifestd_test_leaves_isomorphic_asexual(
        df_a, df_b, "taxon_label"
    )


def test_taxon_label_id_keyword():
    """When taxon_label == 'id', the id column is used directly."""
    df = pd.read_csv(f"{assets_path}/nk_ecoeaselection.csv")
    df_pl = pl.from_pandas(df)
    assert alifestd_test_leaves_isomorphic_polars(df_pl, df_pl, "id")


def test_negative_different_node_counts():
    """Different node counts should immediately return False."""
    df_a = alifestd_make_leaf_split(n_leaves=8, seed=0)
    df_b = alifestd_make_leaf_split(n_leaves=16, seed=0)
    df_a["taxon_label"] = df_a["id"].astype(str)
    df_b["taxon_label"] = df_b["id"].astype(str)
    assert not alifestd_test_leaves_isomorphic_polars(
        pl.from_pandas(df_a), pl.from_pandas(df_b), "taxon_label"
    )
