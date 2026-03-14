import os

import pandas as pd
import pytest

from phyloframe.legacy import (
    alifestd_find_leaf_ids,
)
from phyloframe.legacy import (
    alifestd_ladderize_asexual as alifestd_ladderize_asexual_,
)
from phyloframe.legacy import (
    alifestd_make_empty,
    alifestd_validate,
)

from ._impl import enforce_dtype_stability_pandas

alifestd_ladderize_asexual = enforce_dtype_stability_pandas(
    alifestd_ladderize_asexual_
)

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

    result = alifestd_ladderize_asexual(phylogeny_df)

    assert alifestd_validate(result)
    assert original.equals(phylogeny_df)

    # same set of ids
    assert set(result["id"]) == set(phylogeny_df["id"])
    # same number of rows
    assert len(result) == len(phylogeny_df)
    # same leaf ids
    assert set(alifestd_find_leaf_ids(result)) == set(
        alifestd_find_leaf_ids(phylogeny_df)
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
    ],
)
def test_fuzz_reverse(phylogeny_df: pd.DataFrame):
    original = phylogeny_df.copy()

    result = alifestd_ladderize_asexual(phylogeny_df, reverse=True)

    assert alifestd_validate(result)
    assert original.equals(phylogeny_df)

    assert set(result["id"]) == set(phylogeny_df["id"])
    assert len(result) == len(phylogeny_df)


def test_empty():
    res = alifestd_ladderize_asexual(alifestd_make_empty())
    assert len(res) == 0


@pytest.mark.parametrize("mutate", [True, False])
def test_simple_chain(mutate: bool):
    """Chain: 0 -> 1 -> 2. No reordering needed."""
    phylogeny_df = pd.DataFrame(
        {
            "id": [0, 1, 2],
            "ancestor_list": ["[None]", "[0]", "[1]"],
        }
    )
    original_df = phylogeny_df.copy()
    result_df = alifestd_ladderize_asexual(phylogeny_df, mutate=mutate)

    assert list(result_df["id"]) == [0, 1, 2]
    if not mutate:
        assert original_df.equals(phylogeny_df)


@pytest.mark.parametrize("mutate", [True, False])
def test_simple_tree(mutate: bool):
    """Tree structure:
        0 (root)
        +-- 1
        |   +-- 3
        |   +-- 4
        +-- 2

    Node 2 has 1 leaf (itself), node 1 has 2 leaves (3, 4).
    Ladderized: 2 (fewer leaves) should come before 1's subtree.
    Expected order: 0, 2, 1, 3, 4 or 0, 2, 1, 4, 3
    """
    phylogeny_df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3, 4],
            "ancestor_list": ["[None]", "[0]", "[0]", "[1]", "[1]"],
        }
    )
    original_df = phylogeny_df.copy()
    result_df = alifestd_ladderize_asexual(phylogeny_df, mutate=mutate)

    result_ids = list(result_df["id"])
    # root first
    assert result_ids[0] == 0
    # node 2 (1 leaf) before node 1 (2 leaves)
    assert result_ids.index(2) < result_ids.index(1)
    # node 1 before its children
    assert result_ids.index(1) < result_ids.index(3)
    assert result_ids.index(1) < result_ids.index(4)

    if not mutate:
        assert original_df.equals(phylogeny_df)


@pytest.mark.parametrize("mutate", [True, False])
def test_simple_tree_reverse(mutate: bool):
    """Same tree, reverse=True: more leaves first.
    Node 1 (2 leaves) should come before node 2 (1 leaf).
    """
    phylogeny_df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3, 4],
            "ancestor_list": ["[None]", "[0]", "[0]", "[1]", "[1]"],
        }
    )
    original_df = phylogeny_df.copy()
    result_df = alifestd_ladderize_asexual(
        phylogeny_df, reverse=True, mutate=mutate
    )

    result_ids = list(result_df["id"])
    assert result_ids[0] == 0
    # node 1 (2 leaves) before node 2 (1 leaf)
    assert result_ids.index(1) < result_ids.index(2)
    assert result_ids.index(1) < result_ids.index(3)
    assert result_ids.index(1) < result_ids.index(4)

    if not mutate:
        assert original_df.equals(phylogeny_df)


@pytest.mark.parametrize("mutate", [True, False])
def test_asymmetric_tree(mutate: bool):
    """Tree structure:
        0
        +-- 1
        |   +-- 3
        |   |   +-- 5
        |   |   +-- 6
        |   +-- 4
        +-- 2

    num_leaves: 0=3, 1=2 (leaves 5,6 via 3; leaf 4), 2=1, 3=2, 4=1, 5=1, 6=1
    Ladderized at root: 2 (1 leaf) before 1 (3 leaves)
    Ladderized at node 1: 4 (1 leaf) before 3 (2 leaves)
    """
    phylogeny_df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3, 4, 5, 6],
            "ancestor_list": [
                "[None]",
                "[0]",
                "[0]",
                "[1]",
                "[1]",
                "[3]",
                "[3]",
            ],
        }
    )
    original_df = phylogeny_df.copy()
    result_df = alifestd_ladderize_asexual(phylogeny_df, mutate=mutate)

    result_ids = list(result_df["id"])
    assert result_ids[0] == 0
    # at root: 2 (1 leaf) before 1 (3 leaves)
    assert result_ids.index(2) < result_ids.index(1)
    # at node 1: 4 (1 leaf) before 3 (2 leaves)
    assert result_ids.index(4) < result_ids.index(3)

    if not mutate:
        assert original_df.equals(phylogeny_df)


@pytest.mark.parametrize("mutate", [True, False])
def test_non_contiguous_ids(mutate: bool):
    """Test with non-contiguous ids (slow path)."""
    phylogeny_df = pd.DataFrame(
        {
            "id": [10, 20, 30, 40, 50],
            "ancestor_list": [
                "[None]",
                "[10]",
                "[10]",
                "[20]",
                "[20]",
            ],
        }
    )
    original_df = phylogeny_df.copy()
    result_df = alifestd_ladderize_asexual(phylogeny_df, mutate=mutate)

    result_ids = list(result_df["id"])
    assert result_ids[0] == 10
    # node 30 (1 leaf) before node 20 (2 leaves)
    assert result_ids.index(30) < result_ids.index(20)

    if not mutate:
        assert original_df.equals(phylogeny_df)


def test_idempotent():
    """Ladderizing an already-ladderized tree should give the same result."""
    phylogeny_df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3, 4],
            "ancestor_list": ["[None]", "[0]", "[0]", "[1]", "[1]"],
        }
    )
    result1 = alifestd_ladderize_asexual(phylogeny_df)
    result2 = alifestd_ladderize_asexual(result1)
    assert list(result1["id"]) == list(result2["id"])


def test_two_roots():
    """Two independent roots."""
    phylogeny_df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "ancestor_list": ["[None]", "[None]", "[0]", "[1]"],
        }
    )
    result = alifestd_ladderize_asexual(phylogeny_df)
    assert set(result["id"]) == {0, 1, 2, 3}
    assert len(result) == 4


def test_single_node():
    """A single root node."""
    phylogeny_df = pd.DataFrame(
        {
            "id": [0],
            "ancestor_list": ["[None]"],
        }
    )
    result = alifestd_ladderize_asexual(phylogeny_df)
    assert list(result["id"]) == [0]
