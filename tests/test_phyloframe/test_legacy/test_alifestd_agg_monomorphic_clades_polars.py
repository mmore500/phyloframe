import os
import typing

import numpy as np
import pandas as pd
import polars as pl
import pytest

from phyloframe.legacy import (
    alifestd_agg_monomorphic_clades_polars as alifestd_agg_monomorphic_clades_polars_,
)
from phyloframe.legacy import (
    alifestd_aggregate_phylogenies,
    alifestd_mark_leaves,
    alifestd_to_working_format,
)

from ._impl import enforce_dtype_stability_polars

alifestd_agg_monomorphic_clades_polars = enforce_dtype_stability_polars(
    alifestd_agg_monomorphic_clades_polars_
)

assets_path = os.path.join(os.path.dirname(__file__), "assets")


def _add_inherited_traits(
    phylogeny_df: pd.DataFrame, seed: int = 1
) -> pd.DataFrame:
    """Add trait columns that are inherited with occasional mutation.

    Requires working format.
    """
    rng = np.random.default_rng(seed)
    ancestor_ids = phylogeny_df["ancestor_id"].to_numpy()
    trait_a = np.zeros(len(phylogeny_df), dtype=np.int64)
    trait_b = np.zeros(len(phylogeny_df), dtype=object)
    trait_b[:] = "x"
    for idx, ancestor_id in enumerate(ancestor_ids):
        if idx == ancestor_id:
            continue
        trait_a[idx] = trait_a[ancestor_id] + (rng.random() < 0.05)
        trait_b[idx] = (
            rng.choice(["x", "y"])
            if rng.random() < 0.05
            else trait_b[ancestor_id]
        )
    phylogeny_df["trait_a"] = trait_a
    phylogeny_df["trait_b"] = trait_b.astype(str)
    return alifestd_mark_leaves(phylogeny_df, mutate=True)


def _reference_groups(
    phylogeny_df: pd.DataFrame, by: list, trait_mask: str = None
) -> dict:
    """Brute-force reference implementation, mapping id to group members."""
    ancestor_of = dict(zip(phylogeny_df["id"], phylogeny_df["ancestor_id"]))
    values = dict(zip(phylogeny_df["id"], zip(*map(phylogeny_df.get, by))))
    considered = (
        dict.fromkeys(phylogeny_df["id"], True)
        if trait_mask is None
        else dict(zip(phylogeny_df["id"], phylogeny_df[trait_mask]))
    )

    def lineage(id_):
        while True:
            yield id_
            if ancestor_of[id_] == id_:
                break
            id_ = ancestor_of[id_]

    clade_members = {id_: set() for id_ in ancestor_of}
    for id_ in ancestor_of:
        for lineage_id in lineage(id_):
            clade_members[lineage_id].add(id_)

    is_monomorphic = {
        id_: len({values[m] for m in members if considered[m]}) <= 1
        for id_, members in clade_members.items()
    }

    groups = {}
    for id_ in ancestor_of:
        group_root = id_
        for lineage_id in lineage(id_):
            if is_monomorphic[lineage_id]:
                group_root = lineage_id
        groups[id_] = (
            clade_members[group_root] if is_monomorphic[group_root] else {id_}
        )
    return groups


def _make_toy_df() -> pl.DataFrame:
    #        0 (a)
    #      /       \
    #   1 (a)      2 (b)
    #   /   \      /   \
    # 3(a) 4(a)  5(b)  6(c)
    #             |
    #            7(b)
    return pl.DataFrame(
        {
            "id": [0, 1, 2, 3, 4, 5, 6, 7],
            "ancestor_id": [0, 0, 0, 1, 1, 2, 2, 5],
            "trait": ["a", "a", "b", "a", "a", "b", "c", "b"],
            "is_leaf": [False, False, False, True, True, False, True, True],
            "origin_time": [0, 1, 1, 2, 3, 2, 2, 4],
        }
    )


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_agg_monomorphic_clades_polars_toy(apply: typing.Callable):
    df = _make_toy_df()
    res = (
        alifestd_agg_monomorphic_clades_polars(
            apply(df),
            by="trait",
            aggs=[pl.col("origin_time").max()],
            named_aggs={
                "clade_size": pl.col("is_leaf").sum(),
                "origin_time_min": pl.col("origin_time").min(),
            },
        )
        .lazy()
        .collect()
    )
    assert res.columns == [*df.columns, "clade_size", "origin_time_min"]
    assert res["id"].to_list() == [0, 1, 2, 5, 6]
    assert res["ancestor_id"].to_list() == [0, 0, 0, 2, 2]
    assert res["trait"].to_list() == ["a", "a", "b", "b", "c"]
    assert res["clade_size"].to_list() == [0, 2, 0, 1, 1]
    assert res["origin_time"].to_list() == [0, 3, 1, 4, 2]
    assert res["origin_time_min"].to_list() == [0, 1, 1, 2, 2]


def test_alifestd_agg_monomorphic_clades_polars_no_aggs():
    df = _make_toy_df()
    res = alifestd_agg_monomorphic_clades_polars(df, by="trait")
    assert res.equals(df.filter(pl.col("id").is_in([0, 1, 2, 5, 6])))


def test_alifestd_agg_monomorphic_clades_polars_unaggregated():
    df = _make_toy_df().with_columns(label=pl.Series(list("ABCDEFGH")))
    res = alifestd_agg_monomorphic_clades_polars(df, by="trait")
    # unaggregated columns retain group root's values
    assert res["label"].to_list() == list("ABCFG")


def test_alifestd_agg_monomorphic_clades_polars_trait_mask():
    df = _make_toy_df().with_columns(
        trait=pl.when(pl.col("id").is_in([0, 1, 5]))
        .then(pl.lit("z"))
        .otherwise(pl.col("trait")),
    )
    res = alifestd_agg_monomorphic_clades_polars(
        df,
        by="trait",
        trait_mask="is_leaf",
        named_aggs={"clade_size": pl.col("is_leaf").sum()},
    )
    assert res["id"].to_list() == [0, 1, 2, 5, 6]
    assert res["clade_size"].to_list() == [0, 2, 0, 1, 1]
    # by columns take group's shared value among marked members
    assert res["trait"].to_list() == ["z", "a", "b", "b", "c"]

    res = alifestd_agg_monomorphic_clades_polars(df, by="trait")
    assert res["id"].to_list() == [0, 1, 2, 3, 4, 5, 6, 7]


def test_alifestd_agg_monomorphic_clades_polars_trait_mask_unmarked_clade():
    df = _make_toy_df().with_columns(
        considered=pl.col("id").is_in([0, 2, 5, 6, 7]),
        trait=pl.when(pl.col("id") == 3)
        .then(pl.lit("q"))
        .when(pl.col("id") == 4)
        .then(pl.lit("r"))
        .otherwise(pl.col("trait")),
    )
    # clade rooted at 1 has no considered members, so is a wildcard...
    res = alifestd_agg_monomorphic_clades_polars(
        df, by="trait", trait_mask="considered"
    )
    assert res["id"].to_list() == [0, 1, 2, 5, 6]
    assert res["trait"].to_list() == ["a", "a", "b", "b", "c"]

    # ... and does not prevent collapse of a parent clade
    res = alifestd_agg_monomorphic_clades_polars(
        df.with_columns(trait=pl.lit("a")), by="trait", trait_mask="considered"
    )
    assert res["id"].to_list() == [0]


def test_alifestd_agg_monomorphic_clades_polars_trait_mask_none_marked():
    df = _make_toy_df().with_columns(considered=pl.lit(False))
    res = alifestd_agg_monomorphic_clades_polars(
        df, by="trait", trait_mask="considered"
    )
    # single group with no marked members takes root's value
    assert res["id"].to_list() == [0]
    assert res["trait"].to_list() == ["a"]


def test_alifestd_agg_monomorphic_clades_polars_override_by_agg():
    df = _make_toy_df()
    res = alifestd_agg_monomorphic_clades_polars(
        df, by="trait", named_aggs={"trait": pl.col("trait").n_unique()}
    )
    assert res["trait"].to_list() == [1] * 5


def test_alifestd_agg_monomorphic_clades_polars_multiple_by():
    df = _make_toy_df().with_columns(
        other=pl.Series([0, 0, 0, 0, 1, 0, 0, 0]),
    )
    res = alifestd_agg_monomorphic_clades_polars(df, by=["trait", "other"])
    assert res["id"].to_list() == [0, 1, 2, 3, 4, 5, 6]


def test_alifestd_agg_monomorphic_clades_polars_nulls():
    df = _make_toy_df().with_columns(
        trait=pl.when(pl.col("id").is_in([1, 3, 4]))
        .then(pl.lit(None))
        .otherwise(pl.col("trait")),
    )
    res = alifestd_agg_monomorphic_clades_polars(df, by="trait")
    assert res["id"].to_list() == [0, 1, 2, 5, 6]


def test_alifestd_agg_monomorphic_clades_polars_all_same():
    df = _make_toy_df().with_columns(trait=pl.lit("a"))
    res = alifestd_agg_monomorphic_clades_polars(
        df, by="trait", named_aggs={"num_nodes": pl.len()}
    )
    assert res["id"].to_list() == [0]
    assert res["num_nodes"].to_list() == [8]


def test_alifestd_agg_monomorphic_clades_polars_ancestor_list():
    df = _make_toy_df().drop("ancestor_id")
    df = df.with_columns(
        ancestor_list=pl.Series(
            ["[none]", "[0]", "[0]", "[1]", "[1]", "[2]", "[2]", "[5]"],
        ),
    )
    res = alifestd_agg_monomorphic_clades_polars(df, by="trait")
    assert res["id"].to_list() == [0, 1, 2, 5, 6]
    assert res["ancestor_list"].to_list() == [
        "[none]",
        "[0]",
        "[0]",
        "[2]",
        "[2]",
    ]


def test_alifestd_agg_monomorphic_clades_polars_empty():
    df = _make_toy_df().head(0)
    res = alifestd_agg_monomorphic_clades_polars(
        df,
        by="trait",
        trait_mask="is_leaf",
        named_aggs={"origin_time_max": pl.col("origin_time").max()},
    )
    assert res.is_empty()
    assert res.columns == [*df.columns, "origin_time_max"]


@pytest.mark.parametrize("col", ["missing", None])
def test_alifestd_agg_monomorphic_clades_polars_missing_column(col):
    df = _make_toy_df()
    with pytest.raises(ValueError):
        alifestd_agg_monomorphic_clades_polars(
            df,
            by="trait" if col is None else col,
            trait_mask="missing" if col is None else None,
        )


@pytest.mark.parametrize("col", ["id", "ancestor_id", "ancestor_list"])
def test_alifestd_agg_monomorphic_clades_polars_protected_column(col):
    df = _make_toy_df().with_columns(
        ancestor_list=pl.format("[{}]", pl.col("ancestor_id")),
    )
    with pytest.raises(ValueError):
        alifestd_agg_monomorphic_clades_polars(
            df, by="trait", aggs=[pl.col(col).first()]
        )


def test_alifestd_agg_monomorphic_clades_polars_noncontiguous():
    df = _make_toy_df().with_columns(
        id=pl.col("id") * 10 + 3,
        ancestor_id=pl.col("ancestor_id") * 10 + 3,
    )
    with pytest.raises(NotImplementedError):
        alifestd_agg_monomorphic_clades_polars(df, by="trait")


def test_alifestd_agg_monomorphic_clades_polars_unsorted():
    df = pl.DataFrame(
        {
            "id": [0, 1, 2],
            "ancestor_id": [0, 2, 0],
            "trait": ["a", "a", "a"],
        }
    )
    with pytest.raises(NotImplementedError):
        alifestd_agg_monomorphic_clades_polars(df, by="trait")


@pytest.mark.parametrize(
    "phylogeny_df",
    [
        pd.read_csv(f"{assets_path}/nk_ecoeaselection.csv"),
        pd.read_csv(f"{assets_path}/nk_lexicaseselection.csv"),
        pd.read_csv(f"{assets_path}/nk_tournamentselection.csv"),
        alifestd_aggregate_phylogenies(
            [
                pd.read_csv(f"{assets_path}/nk_ecoeaselection.csv"),
                pd.read_csv(f"{assets_path}/nk_lexicaseselection.csv"),
            ]
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
@pytest.mark.parametrize(
    "by", [["trait_a"], ["trait_b"], ["trait_a", "trait_b"]]
)
@pytest.mark.parametrize("trait_mask", [None, "is_leaf"])
def test_alifestd_agg_monomorphic_clades_polars_reference(
    phylogeny_df, apply: typing.Callable, by, trait_mask
):
    phylogeny_df = _add_inherited_traits(
        alifestd_to_working_format(phylogeny_df)
    ).drop(columns=["ancestor_list", "phenotype"], errors="ignore")

    res = (
        alifestd_agg_monomorphic_clades_polars(
            apply(pl.from_pandas(phylogeny_df)),
            by=by,
            trait_mask=trait_mask,
            named_aggs={
                "clade_size": pl.col("is_leaf").sum(),
                "origin_time_max": pl.col("origin_time").max(),
            },
        )
        .lazy()
        .collect()
    )

    groups = _reference_groups(phylogeny_df, by, trait_mask)
    ids = phylogeny_df["id"]
    kept_ids = [id_ for id_ in ids if min(groups[id_]) == id_]
    assert 0 < len(kept_ids) < len(ids)
    assert res["id"].to_list() == kept_ids
    assert res["ancestor_id"].to_list() == [
        phylogeny_df["ancestor_id"].iat[id_] for id_ in kept_ids
    ]

    origin_times = dict(zip(ids, phylogeny_df["origin_time"]))
    is_leaf = dict(zip(ids, phylogeny_df["is_leaf"]))
    considered = dict(
        zip(ids, phylogeny_df[trait_mask] if trait_mask else [True] * len(ids))
    )
    assert res["clade_size"].to_list() == [
        sum(map(is_leaf.get, groups[id_])) for id_ in kept_ids
    ]
    assert res["clade_size"].sum() == phylogeny_df["is_leaf"].sum()
    assert res["origin_time_max"].to_list() == [
        max(map(origin_times.get, groups[id_])) for id_ in kept_ids
    ]
    assert res["origin_time"].to_list() == [
        origin_times[id_] for id_ in kept_ids
    ]
    for col in by:
        values = dict(zip(ids, phylogeny_df[col]))
        expected = []
        for id_ in kept_ids:
            group = sorted(groups[id_])
            group_considered = [m for m in group if considered[m]] or group
            expected.append(values[group_considered[0]])
        assert res[col].to_list() == expected
