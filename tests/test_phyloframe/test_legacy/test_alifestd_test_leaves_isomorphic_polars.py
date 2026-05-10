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
    alifestd_to_working_format,
)
from phyloframe.legacy._alifestd_test_leaves_isomorphic_polars import (
    alifestd_test_leaves_isomorphic_polars,
)

assets_path = os.path.join(os.path.dirname(__file__), "assets")


def _prepare_pl(phylogeny_df: pd.DataFrame) -> pl.DataFrame:
    """Pre-canonicalize a pandas phylogeny and convert to polars."""
    return pl.from_pandas(alifestd_to_working_format(phylogeny_df))


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
    original_df = _prepare_pl(
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
    df_pl = _prepare_pl(phylogeny_df)

    assert alifestd_test_leaves_isomorphic_polars(df_pl, df_pl, "taxon_label")
    assert alifestd_test_leaves_isomorphic_polars(
        df_pl,
        _prepare_pl(alifestd_collapse_unifurcations(phylogeny_df)),
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
    df_pl = _prepare_pl(phylogeny_df)

    assert alifestd_test_leaves_isomorphic_polars(df_pl, df_pl, "taxon_label")
    assert alifestd_test_leaves_isomorphic_polars(
        df_pl,
        _prepare_pl(alifestd_collapse_unifurcations(phylogeny_df)),
        "taxon_label",
    )


def test_negative_different_topologies():
    df1 = pd.read_csv(f"{assets_path}/nk_lexicaseselection.csv")
    df2 = pd.read_csv(f"{assets_path}/nk_ecoeaselection.csv")
    df1["taxon_label"] = df1["id"]
    df2["taxon_label"] = df2["id"]
    assert not alifestd_test_leaves_isomorphic_polars(
        _prepare_pl(df1), _prepare_pl(df2), "taxon_label"
    )


def test_negative_tweaked():
    df1 = pd.read_csv(f"{assets_path}/nk_ecoeaselection.csv")
    df2 = pd.read_csv(f"{assets_path}/nk_ecoeaselection_tweaked.csv")
    df1["taxon_label"] = df1["id"].astype(str)
    df2["taxon_label"] = df2["id"].astype(str)
    assert not alifestd_test_leaves_isomorphic_polars(
        _prepare_pl(df1), _prepare_pl(df2), "taxon_label"
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
    """Fuzz: polars result should agree with pandas asexual implementation."""
    phylogeny_df = phylogeny_df.copy()
    phylogeny_df["taxon_label"] = phylogeny_df["id"].astype(str)
    df_pl = _prepare_pl(phylogeny_df)

    # self-comparison
    assert alifestd_test_leaves_isomorphic_polars(
        df_pl, df_pl, "taxon_label"
    ) == alifestd_test_leaves_isomorphic_asexual(
        phylogeny_df, phylogeny_df, "taxon_label"
    )

    # collapsed-vs-original
    collapsed_pd = alifestd_collapse_unifurcations(phylogeny_df)
    collapsed_pd["taxon_label"] = collapsed_pd["id"].astype(str)
    collapsed_pl = _prepare_pl(collapsed_pd)
    assert alifestd_test_leaves_isomorphic_polars(
        df_pl, collapsed_pl, "taxon_label"
    ) == alifestd_test_leaves_isomorphic_asexual(
        phylogeny_df, collapsed_pd, "taxon_label"
    )


@pytest.mark.parametrize(
    "seed",
    [0, 1, 2, 7, 42, 100, 314, 2718],
)
def test_fuzz_random_trees_positive(seed: int):
    """A tree compared against itself should be isomorphic."""
    df_a = alifestd_make_leaf_split(n_leaves=24, seed=seed)
    df_a["taxon_label"] = df_a["id"].astype(str)

    df_a_pl = _prepare_pl(df_a)
    assert alifestd_test_leaves_isomorphic_polars(
        df_a_pl, df_a_pl, "taxon_label"
    )
    assert alifestd_test_leaves_isomorphic_asexual(df_a, df_a, "taxon_label")


@pytest.mark.parametrize(
    "seed_pair",
    [(0, 1), (2, 3), (4, 5), (7, 42), (100, 314)],
)
def test_fuzz_random_trees_negative(seed_pair: tuple):
    """Two independently-sampled Yule trees with the same labeled leaf set
    are very unlikely to be isomorphic, but the polars result must still
    agree with the pandas reference.
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
    assert len(leaves_a) == len(leaves_b) == 20

    df_a = df_a.copy()
    df_b = df_b.copy()
    df_a["taxon_label"] = "x"
    df_b["taxon_label"] = "x"
    for new, old in zip(sorted(leaves_a), range(20)):
        df_a.loc[df_a["id"] == new, "taxon_label"] = f"leaf_{old}"
    for new, old in zip(sorted(leaves_b), range(20)):
        df_b.loc[df_b["id"] == new, "taxon_label"] = f"leaf_{old}"

    pl_a = _prepare_pl(df_a)
    pl_b = _prepare_pl(df_b)

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
    assert not alifestd_test_leaves_isomorphic_polars(
        _prepare_pl(df_a), _prepare_pl(df_b), "taxon_label"
    )
    assert not alifestd_test_leaves_isomorphic_asexual(
        df_a, df_b, "taxon_label"
    )


def test_taxon_label_id_keyword():
    """When taxon_label == 'id', the id column is used directly."""
    df = pd.read_csv(f"{assets_path}/nk_ecoeaselection.csv")
    df_pl = _prepare_pl(df)
    assert alifestd_test_leaves_isomorphic_polars(df_pl, df_pl, "id")


def test_negative_different_node_counts():
    """Different node counts should immediately return False."""
    df_a = alifestd_make_leaf_split(n_leaves=8, seed=0)
    df_b = alifestd_make_leaf_split(n_leaves=16, seed=0)
    df_a["taxon_label"] = df_a["id"].astype(str)
    df_b["taxon_label"] = df_b["id"].astype(str)
    assert not alifestd_test_leaves_isomorphic_polars(
        _prepare_pl(df_a), _prepare_pl(df_b), "taxon_label"
    )


def test_raises_on_unsorted_input():
    """Non-topologically-sorted input should raise NotImplementedError."""
    df = _prepare_pl(pd.read_csv(f"{assets_path}/nk_ecoeaselection.csv"))
    df = df.with_columns(taxon_label=pl.col("id"))
    shuffled = df.sample(fraction=1.0, shuffle=True, seed=1)
    with pytest.raises(NotImplementedError):
        alifestd_test_leaves_isomorphic_polars(df, shuffled, "taxon_label")


def test_raises_on_non_contiguous_ids():
    """Non-contiguous ids should raise NotImplementedError."""
    df = pl.DataFrame(
        {
            "id": [0, 2, 4],
            "ancestor_id": [0, 0, 2],
            "taxon_label": ["a", "b", "c"],
        }
    )
    with pytest.raises(NotImplementedError):
        alifestd_test_leaves_isomorphic_polars(df, df, "taxon_label")


def test_ancestor_id_derived_from_ancestor_list():
    """If only ``ancestor_list`` is present, ancestor_id should be derived."""
    df = pl.DataFrame(
        {
            "id": [0, 1, 2],
            "ancestor_list": ["[none]", "[0]", "[0]"],
            "taxon_label": ["a", "b", "c"],
        }
    )
    assert alifestd_test_leaves_isomorphic_polars(df, df, "taxon_label")


def test_ancestor_list_silently_dropped():
    """Presence of ancestor_list should not prevent the comparison."""
    df = pl.DataFrame(
        {
            "id": [0, 1, 2],
            "ancestor_id": [0, 0, 0],
            "ancestor_list": ["[none]", "[0]", "[0]"],
            "taxon_label": ["a", "b", "c"],
        }
    )
    assert alifestd_test_leaves_isomorphic_polars(df, df, "taxon_label")
