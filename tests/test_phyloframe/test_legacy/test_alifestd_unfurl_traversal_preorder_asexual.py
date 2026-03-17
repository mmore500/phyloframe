import os

import pandas as pd
import pytest

from phyloframe.legacy import (
    alifestd_find_leaf_ids,
    alifestd_find_root_ids,
    alifestd_make_empty,
)
from phyloframe.legacy import (
    alifestd_unfurl_traversal_preorder_asexual as alifestd_unfurl_traversal_preorder_asexual_,
)

from ._impl import enforce_dtype_stability_pandas

alifestd_unfurl_traversal_preorder_asexual = enforce_dtype_stability_pandas(
    alifestd_unfurl_traversal_preorder_asexual_
)


def is_nonincreasing(seq):
    return all(a >= b for a, b in zip(seq, seq[1:]))


assets_path = os.path.join(os.path.dirname(__file__), "assets")


@pytest.mark.parametrize(
    "phylogeny_df",
    [
        pd.read_csv(
            f"{assets_path}/example-standard-toy-asexual-phylogeny.csv"
        ),
        pd.read_csv(f"{assets_path}/nk_ecoeaselection.csv"),
        pd.read_csv(f"{assets_path}/nk_lexicaseselection.csv"),
        pd.read_csv(f"{assets_path}/nk_tournamentselection.csv"),
    ],
)
def test_fuzz(phylogeny_df: pd.DataFrame):
    original = phylogeny_df.copy()

    result = alifestd_unfurl_traversal_preorder_asexual(phylogeny_df)

    assert original.equals(phylogeny_df)

    assert len(result) == len(phylogeny_df)
    assert set(result) == set(phylogeny_df["id"])

    # In preorder, first element is a root and last is a leaf
    assert result[0] in alifestd_find_root_ids(phylogeny_df)
    assert result[-1] in alifestd_find_leaf_ids(phylogeny_df)

    # Verify valid preorder: every parent appears before its children
    from phyloframe.legacy import alifestd_try_add_ancestor_id_col

    ref_df = alifestd_try_add_ancestor_id_col(phylogeny_df.copy())
    ancestor_of = dict(zip(ref_df["id"], ref_df["ancestor_id"]))
    pos = {node: i for i, node in enumerate(result)}
    for node_id in result:
        ancestor_id = ancestor_of[node_id]
        if node_id != ancestor_id:
            assert pos[ancestor_id] < pos[node_id]


def test_empty():
    res = alifestd_unfurl_traversal_preorder_asexual(alifestd_make_empty())
    assert len(res) == 0


@pytest.mark.parametrize("mutate", [True, False])
def test_simple1(mutate: bool):
    phylogeny_df = pd.DataFrame(
        {
            "id": [0, 1, 2],
            "ancestor_list": ["[None]", "[0]", "[1]"],
        },
    )
    original_df = phylogeny_df.copy()
    result = alifestd_unfurl_traversal_preorder_asexual(
        phylogeny_df,
        mutate=mutate,
    )
    assert result.tolist() == [0, 1, 2]

    if not mutate:
        assert original_df.equals(phylogeny_df)


@pytest.mark.parametrize("mutate", [True, False])
def test_simple2(mutate: bool):
    phylogeny_df = pd.DataFrame(
        {
            "id": [3, 0, 1, 2],
            "ancestor_list": ["[0]", "[1]", "[None]", "[1]"],
        },
    )
    original_df = phylogeny_df.copy()
    result = alifestd_unfurl_traversal_preorder_asexual(
        phylogeny_df,
        mutate=mutate,
    )
    # Tree: 1 is root, children are 0 and 2; 0 has child 3
    # Preorder: 1, 0, 3, 2
    assert result.tolist() == [1, 0, 3, 2]

    if not mutate:
        assert original_df.equals(phylogeny_df)


@pytest.mark.parametrize("mutate", [True, False])
def test_simple3(mutate: bool):
    phylogeny_df = pd.DataFrame(
        {
            "id": [4, 0, 2, 3],
            "ancestor_list": ["[None]", "[None]", "[0]", "[4]"],
        },
    )
    original_df = phylogeny_df.copy()
    result = alifestd_unfurl_traversal_preorder_asexual(
        phylogeny_df,
        mutate=mutate,
    )
    # Two roots: 0 (child 2) and 4 (child 3)
    # Preorder: 4, 3, 0, 2 (root ordering follows topological sort)
    assert result.tolist() == [4, 3, 0, 2]

    if not mutate:
        assert original_df.equals(phylogeny_df)


@pytest.mark.parametrize("mutate", [True, False])
def test_simple4(mutate: bool):
    phylogeny_df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "ancestor_list": ["[None]", "[0]", "[0]", "[1]"],
        },
    )
    original_df = phylogeny_df.copy()
    result = alifestd_unfurl_traversal_preorder_asexual(
        phylogeny_df,
        mutate=mutate,
    )
    # Tree: 0 -> {1, 2}, 1 -> {3}
    # Preorder: 0, 1, 3, 2
    assert result.tolist() == [0, 1, 3, 2]

    if not mutate:
        assert original_df.equals(phylogeny_df)


def test_with_num_children_col():
    """Test that pre-existing num_children column is reused on fast path."""
    phylogeny_df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "ancestor_id": [0, 0, 0, 1],
            "num_children": [2, 1, 0, 0],
        },
    )
    result = alifestd_unfurl_traversal_preorder_asexual(phylogeny_df)
    assert result.tolist() == [0, 1, 3, 2]


def test_with_sibling_cols():
    """Test sibling fast path with first_child_id + next_sibling_id."""
    phylogeny_df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "ancestor_id": [0, 0, 0, 1],
            "first_child_id": [1, 3, 2, 3],
            "next_sibling_id": [0, 2, 2, 3],
        },
    )
    result = alifestd_unfurl_traversal_preorder_asexual(phylogeny_df)
    assert result.tolist() == [0, 1, 3, 2]


@pytest.mark.parametrize(
    "phylogeny_df",
    [
        pd.read_csv(
            f"{assets_path}/example-standard-toy-asexual-phylogeny.csv"
        ),
        pd.read_csv(f"{assets_path}/nk_ecoeaselection.csv"),
        pd.read_csv(f"{assets_path}/nk_lexicaseselection.csv"),
        pd.read_csv(f"{assets_path}/nk_tournamentselection.csv"),
    ],
)
def test_sibling_fast_path_matches_baseline(phylogeny_df: pd.DataFrame):
    """Verify sibling-based fast path gives same result as CSR-based."""
    from phyloframe.legacy import (
        alifestd_mark_first_child_id_asexual,
        alifestd_mark_next_sibling_id_asexual,
        alifestd_try_add_ancestor_id_col,
    )

    result_base = alifestd_unfurl_traversal_preorder_asexual(phylogeny_df)

    df_sib = alifestd_try_add_ancestor_id_col(phylogeny_df.copy())
    df_sib = alifestd_mark_first_child_id_asexual(df_sib, mutate=True)
    df_sib = alifestd_mark_next_sibling_id_asexual(df_sib, mutate=True)
    result_sib = alifestd_unfurl_traversal_preorder_asexual(df_sib)

    assert result_base.tolist() == result_sib.tolist()
