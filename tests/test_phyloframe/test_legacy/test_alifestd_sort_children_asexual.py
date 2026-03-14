import os

import pandas as pd
import pytest

from phyloframe.legacy import (
    alifestd_find_leaf_ids,
    alifestd_make_empty,
    alifestd_mark_num_leaves_asexual,
)
from phyloframe.legacy import (
    alifestd_sort_children_asexual as alifestd_sort_children_asexual_,
)
from phyloframe.legacy import (
    alifestd_validate,
)

from ._impl import enforce_dtype_stability_pandas

alifestd_sort_children_asexual = enforce_dtype_stability_pandas(
    alifestd_sort_children_asexual_
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

    phylogeny_df = alifestd_mark_num_leaves_asexual(phylogeny_df)
    result = alifestd_sort_children_asexual(
        phylogeny_df,
        criterion="num_leaves",
    )

    assert alifestd_validate(result)
    assert original.equals(original)  # input not mutated

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
    phylogeny_df = alifestd_mark_num_leaves_asexual(phylogeny_df)
    result = alifestd_sort_children_asexual(
        phylogeny_df,
        criterion="num_leaves",
        reverse=True,
    )

    assert alifestd_validate(result)

    assert set(result["id"]) == set(phylogeny_df["id"])
    assert len(result) == len(phylogeny_df)


def test_empty():
    res = alifestd_sort_children_asexual(
        alifestd_make_empty().assign(num_leaves=[]),
        criterion="num_leaves",
    )
    assert len(res) == 0


@pytest.mark.parametrize("mutate", [True, False])
def test_simple_tree(mutate: bool):
    """Tree structure:
        0 (root)
        +-- 1
        |   +-- 3
        |   +-- 4
        +-- 2

    num_leaves: 0=3, 1=2, 2=1, 3=1, 4=1
    Sorted ascending: 2 (1 leaf) before 1's subtree (2 leaves).
    """
    phylogeny_df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3, 4],
            "ancestor_list": ["[None]", "[0]", "[0]", "[1]", "[1]"],
        }
    )
    phylogeny_df = alifestd_mark_num_leaves_asexual(phylogeny_df, mutate=True)
    original_df = phylogeny_df.copy()

    result_df = alifestd_sort_children_asexual(
        phylogeny_df,
        criterion="num_leaves",
        mutate=mutate,
    )

    result_ids = list(result_df["id"])
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
    """Reverse: higher values first."""
    phylogeny_df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3, 4],
            "ancestor_list": ["[None]", "[0]", "[0]", "[1]", "[1]"],
        }
    )
    phylogeny_df = alifestd_mark_num_leaves_asexual(phylogeny_df, mutate=True)

    result_df = alifestd_sort_children_asexual(
        phylogeny_df,
        criterion="num_leaves",
        reverse=True,
        mutate=mutate,
    )

    result_ids = list(result_df["id"])
    assert result_ids[0] == 0
    # node 1 (2 leaves) before node 2 (1 leaf)
    assert result_ids.index(1) < result_ids.index(2)
    assert result_ids.index(1) < result_ids.index(3)
    assert result_ids.index(1) < result_ids.index(4)


def test_single_node():
    phylogeny_df = pd.DataFrame(
        {
            "id": [0],
            "ancestor_list": ["[None]"],
            "num_leaves": [1],
        }
    )
    result = alifestd_sort_children_asexual(
        phylogeny_df,
        criterion="num_leaves",
    )
    assert list(result["id"]) == [0]


def test_chain():
    """Chain: 0 -> 1 -> 2. No reordering needed."""
    phylogeny_df = pd.DataFrame(
        {
            "id": [0, 1, 2],
            "ancestor_list": ["[None]", "[0]", "[1]"],
        }
    )
    phylogeny_df = alifestd_mark_num_leaves_asexual(phylogeny_df, mutate=True)
    result_df = alifestd_sort_children_asexual(
        phylogeny_df,
        criterion="num_leaves",
    )

    assert list(result_df["id"]) == [0, 1, 2]


def test_non_contiguous_ids():
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
    phylogeny_df = alifestd_mark_num_leaves_asexual(phylogeny_df, mutate=True)
    result_df = alifestd_sort_children_asexual(
        phylogeny_df,
        criterion="num_leaves",
    )

    result_ids = list(result_df["id"])
    assert result_ids[0] == 10
    # node 30 (1 leaf) before node 20 (2 leaves)
    assert result_ids.index(30) < result_ids.index(20)


def test_custom_criterion():
    """Sort by a custom criterion column instead of num_leaves."""
    phylogeny_df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3, 4],
            "ancestor_list": ["[None]", "[0]", "[0]", "[1]", "[1]"],
            "priority": [0, 100, 1, 50, 60],
        }
    )
    result_df = alifestd_sort_children_asexual(
        phylogeny_df,
        criterion="priority",
    )

    result_ids = list(result_df["id"])
    assert result_ids[0] == 0
    # node 2 (priority=1) before node 1 (priority=100)
    assert result_ids.index(2) < result_ids.index(1)


def test_two_roots():
    """Two independent roots."""
    phylogeny_df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "ancestor_list": ["[None]", "[None]", "[0]", "[1]"],
            "num_leaves": [1, 1, 1, 1],
        }
    )
    result = alifestd_sort_children_asexual(
        phylogeny_df,
        criterion="num_leaves",
    )
    assert set(result["id"]) == {0, 1, 2, 3}
    assert len(result) == 4


def test_idempotent():
    """Sorting an already-sorted tree should give the same result."""
    phylogeny_df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3, 4],
            "ancestor_list": ["[None]", "[0]", "[0]", "[1]", "[1]"],
        }
    )
    phylogeny_df = alifestd_mark_num_leaves_asexual(phylogeny_df, mutate=True)
    result1 = alifestd_sort_children_asexual(
        phylogeny_df,
        criterion="num_leaves",
    )
    result2 = alifestd_sort_children_asexual(
        result1,
        criterion="num_leaves",
    )
    assert list(result1["id"]) == list(result2["id"])
