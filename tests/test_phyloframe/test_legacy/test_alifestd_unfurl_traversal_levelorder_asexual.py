import os

import pandas as pd
import pytest

from phyloframe.legacy import (
    alifestd_find_leaf_ids,
    alifestd_find_root_ids,
    alifestd_make_empty,
    alifestd_mark_node_depth_asexual,
)
from phyloframe.legacy import (
    alifestd_unfurl_traversal_levelorder_asexual as alifestd_unfurl_traversal_levelorder_asexual_,
)

from ._impl import enforce_dtype_stability_pandas

alifestd_unfurl_traversal_levelorder_asexual = enforce_dtype_stability_pandas(
    alifestd_unfurl_traversal_levelorder_asexual_
)


def is_nondecreasing(seq):
    return all(a <= b for a, b in zip(seq, seq[1:]))


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

    result = alifestd_unfurl_traversal_levelorder_asexual(phylogeny_df)

    assert original.equals(phylogeny_df)

    assert len(result) == len(phylogeny_df)
    assert set(result) == set(phylogeny_df["id"])

    # In levelorder, depths should be nondecreasing
    reference_df = alifestd_mark_node_depth_asexual(phylogeny_df)
    node_depths = dict(
        zip(
            reference_df["id"],
            reference_df["node_depth"],
        ),
    )
    result_depths = [node_depths[id_] for id_ in result]
    assert is_nondecreasing(result_depths)

    # First element is a root and last is a leaf
    assert result[0] in alifestd_find_root_ids(phylogeny_df)
    assert result[-1] in alifestd_find_leaf_ids(phylogeny_df)

    # Verify valid levelorder: every parent appears before its children
    from phyloframe.legacy import alifestd_try_add_ancestor_id_col

    ref_df = alifestd_try_add_ancestor_id_col(phylogeny_df.copy())
    ancestor_of = dict(zip(ref_df["id"], ref_df["ancestor_id"]))
    pos = {node: i for i, node in enumerate(result)}
    for node_id in result:
        ancestor_id = ancestor_of[node_id]
        if node_id != ancestor_id:
            assert pos[ancestor_id] < pos[node_id]


def test_empty():
    res = alifestd_unfurl_traversal_levelorder_asexual(alifestd_make_empty())
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
    result = alifestd_unfurl_traversal_levelorder_asexual(
        phylogeny_df,
        mutate=mutate,
    )
    # Chain: 0 -> 1 -> 2, levelorder same as topological
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
    result = alifestd_unfurl_traversal_levelorder_asexual(
        phylogeny_df,
        mutate=mutate,
    )
    # Tree: 1 is root, children are 0 and 2; 0 has child 3
    # Depths: 1=0, 0=1, 2=1, 3=2
    # Levelorder: 1, 0, 2, 3
    assert result.tolist() == [1, 0, 2, 3]

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
    result = alifestd_unfurl_traversal_levelorder_asexual(
        phylogeny_df,
        mutate=mutate,
    )
    # Two roots: 0 (child 2) and 4 (child 3)
    # Depths: 4=0, 0=0, 2=1, 3=1
    # Levelorder: 4, 0, 2, 3 (roots first, then depth 1)
    assert result.tolist() == [4, 0, 2, 3]

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
    result = alifestd_unfurl_traversal_levelorder_asexual(
        phylogeny_df,
        mutate=mutate,
    )
    # Tree: 0 -> {1, 2}, 1 -> {3}
    # Depths: 0=0, 1=1, 2=1, 3=2
    # Levelorder: 0, 1, 2, 3
    assert result.tolist() == [0, 1, 2, 3]

    if not mutate:
        assert original_df.equals(phylogeny_df)


def test_with_node_depth_col():
    """Test that pre-existing node_depth column is reused on fast path."""
    phylogeny_df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "ancestor_id": [0, 0, 0, 1],
            "node_depth": [0, 1, 1, 2],
        },
    )
    result = alifestd_unfurl_traversal_levelorder_asexual(phylogeny_df)
    assert result.tolist() == [0, 1, 2, 3]
