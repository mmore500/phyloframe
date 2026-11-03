import os

import pandas as pd
import pytest

from phyloframe.legacy import (
    alifestd_find_root_ids,
    alifestd_make_empty,
    alifestd_mark_num_leaves_asexual,
    alifestd_mark_roots,
    alifestd_to_working_format,
    alifestd_try_add_ancestor_id_col,
)
from phyloframe.legacy._alifestd_mark_num_mrca_leaf_pairs_asexual import (
    alifestd_mark_num_mrca_leaf_pairs_asexual as alifestd_mark_num_mrca_leaf_pairs_asexual_,
)

from ._impl import enforce_dtype_stability_pandas

alifestd_mark_num_mrca_leaf_pairs_asexual = enforce_dtype_stability_pandas(
    alifestd_mark_num_mrca_leaf_pairs_asexual_,
)

assets_path = os.path.join(os.path.dirname(__file__), "assets")


@pytest.mark.parametrize(
    "phylogeny_df",
    [
        pd.read_csv(
            f"{assets_path}/example-standard-toy-asexual-phylogeny.csv"
        ),
        pd.read_csv(
            f"{assets_path}/example-standard-toy-asexual-phylogeny-tworoots.csv"
        ),
        pd.read_csv(f"{assets_path}/nk_ecoeaselection.csv"),
        pd.read_csv(f"{assets_path}/nk_lexicaseselection.csv"),
        pd.read_csv(f"{assets_path}/nk_tournamentselection.csv"),
        pd.read_csv(f"{assets_path}/trunktestphylo.csv"),
    ],
)
def test_fuzz(phylogeny_df: pd.DataFrame):
    phylogeny_df = alifestd_try_add_ancestor_id_col(
        alifestd_to_working_format(phylogeny_df), mutate=True
    )
    original = phylogeny_df.copy()

    result = alifestd_mark_num_mrca_leaf_pairs_asexual(phylogeny_df)
    assert original.equals(phylogeny_df)
    assert (result["num_mrca_leaf_pairs"] >= 0).all()

    result = alifestd_mark_num_leaves_asexual(result)
    result = alifestd_try_add_ancestor_id_col(result)
    result = alifestd_mark_roots(result).set_index("id", drop=False)

    # leaf pairs sum to C(N, 2) within each tree
    for root_id in alifestd_find_root_ids(result):
        tree_ids = [root_id]
        stack = [root_id]
        children = result.groupby("ancestor_id")["id"].apply(list)
        while stack:
            id_ = stack.pop()
            for child in children.get(id_, []):
                if child != id_:
                    tree_ids.append(child)
                    stack.append(child)
        num_leaves = result.at[root_id, "num_leaves"]
        assert result.loc[tree_ids, "num_mrca_leaf_pairs"].sum() == (
            num_leaves * (num_leaves - 1) // 2
        )

    # zero for leaves
    assert (
        result.loc[
            ~result["id"].isin(result["ancestor_id"]), "num_mrca_leaf_pairs"
        ]
        == 0
    ).all()


def test_empty():
    res = alifestd_mark_num_mrca_leaf_pairs_asexual(alifestd_make_empty())
    assert "num_mrca_leaf_pairs" in res
    assert len(res) == 0


@pytest.mark.parametrize("mutate", [True, False])
def test_simple(mutate: bool):
    # 0: root polytomy with leaves 1, 2 and inner 3
    # 3: unifurcation to 4
    # 4: bifurcation with leaves 5, 6
    phylogeny_df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3, 4, 5, 6],
            "ancestor_id": [0, 0, 0, 0, 3, 4, 4],
        }
    )
    original_df = phylogeny_df.copy()
    result_df = alifestd_mark_num_mrca_leaf_pairs_asexual(
        phylogeny_df,
        mutate=mutate,
    )
    assert result_df["num_mrca_leaf_pairs"].tolist() == [5, 0, 0, 0, 1, 0, 0]
    assert result_df["num_mrca_leaf_pairs"].sum() == 4 * 3 // 2
    if not mutate:
        assert original_df.equals(phylogeny_df)


def test_multiple_roots():
    phylogeny_df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3, 4, 5, 6],
            "ancestor_list": [
                "[none]",
                "[0]",
                "[0]",
                "[none]",
                "[3]",
                "[3]",
                "[3]",
            ],
        }
    )
    result_df = alifestd_mark_num_mrca_leaf_pairs_asexual(phylogeny_df)
    assert result_df["num_mrca_leaf_pairs"].tolist() == [1, 0, 0, 3, 0, 0, 0]


def test_mark_as():
    phylogeny_df = pd.DataFrame(
        {"id": [0, 1, 2], "ancestor_id": [0, 0, 0]},
    )
    result_df = alifestd_mark_num_mrca_leaf_pairs_asexual(
        phylogeny_df, mark_as="custom"
    )
    assert "num_mrca_leaf_pairs" not in result_df
    assert result_df["custom"].tolist() == [1, 0, 0]


def test_existing_num_leaves_used():
    phylogeny_df = pd.DataFrame(
        {"id": [0, 1, 2], "ancestor_id": [0, 0, 0], "num_leaves": [2, 1, 1]},
    )
    result_df = alifestd_mark_num_mrca_leaf_pairs_asexual(phylogeny_df)
    assert result_df["num_mrca_leaf_pairs"].tolist() == [1, 0, 0]

    # deliberately inconsistent column, to check it is used as-is
    phylogeny_df["num_leaves"] = [3, 1, 1]
    result_df = alifestd_mark_num_mrca_leaf_pairs_asexual(phylogeny_df)
    assert result_df["num_mrca_leaf_pairs"].tolist() == [3, 0, 0]


def test_num_leaves_added():
    phylogeny_df = pd.DataFrame({"id": [0, 1, 2], "ancestor_id": [0, 0, 0]})
    result_df = alifestd_mark_num_mrca_leaf_pairs_asexual(phylogeny_df)
    assert result_df["num_leaves"].tolist() == [2, 1, 1]


@pytest.mark.parametrize(
    "phylogeny_df",
    [
        pd.DataFrame({"id": [0, 1, 3], "ancestor_id": [0, 0, 1]}),
        pd.DataFrame({"id": [0, 1, 2, 3, 4], "ancestor_id": [4, 4, 3, 3, 3]}),
    ],
)
def test_noncontiguous_or_unsorted_unsupported(phylogeny_df: pd.DataFrame):
    with pytest.raises(NotImplementedError):
        alifestd_mark_num_mrca_leaf_pairs_asexual(phylogeny_df)


def test_sexual_unsupported():
    phylogeny_df = pd.DataFrame(
        {
            "id": [0, 1, 2],
            "ancestor_list": ["[none]", "[0]", "[0,1]"],
        },
    )
    with pytest.raises(NotImplementedError):
        alifestd_mark_num_mrca_leaf_pairs_asexual(phylogeny_df)
