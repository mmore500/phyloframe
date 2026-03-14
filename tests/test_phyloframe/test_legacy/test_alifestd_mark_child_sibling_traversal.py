"""Integration tests for tree traversal using first_child_id, next_sibling_id,
and prev_sibling_id columns together."""

from collections import deque
import os

import pandas as pd
import polars as pl
import pytest

from phyloframe.legacy import (
    alifestd_mark_first_child_id_asexual,
    alifestd_mark_next_sibling_id_asexual,
    alifestd_mark_prev_sibling_id_asexual,
    alifestd_to_working_format,
)
from phyloframe.legacy._alifestd_mark_first_child_id_polars import (
    alifestd_mark_first_child_id_polars,
)
from phyloframe.legacy._alifestd_mark_next_sibling_id_polars import (
    alifestd_mark_next_sibling_id_polars,
)
from phyloframe.legacy._alifestd_mark_prev_sibling_id_polars import (
    alifestd_mark_prev_sibling_id_polars,
)

assets_path = os.path.join(os.path.dirname(__file__), "assets")


def _depth_first_pandas(df: pd.DataFrame) -> list:
    """Perform depth-first traversal using first_child and next_sibling."""
    first_child = df["first_child_id"].to_list()
    next_sib = df["next_sibling_id"].to_list()

    visited = []
    stack = [0]  # start from root

    while stack:
        node = stack.pop()
        visited.append(node)

        # push next sibling (to visit after this subtree)
        if next_sib[node] != node:
            stack.append(next_sib[node])

        # push first child (to visit next)
        if first_child[node] != node:
            stack.append(first_child[node])

    return visited


def _breadth_first_pandas(df: pd.DataFrame) -> list:
    """Perform breadth-first traversal using first_child and next_sibling."""
    first_child = df["first_child_id"].to_list()
    next_sib = df["next_sibling_id"].to_list()

    visited = []
    queue = deque([0])

    while queue:
        node = queue.popleft()
        visited.append(node)

        # enqueue all children via first_child + next_sibling chain
        child = first_child[node]
        if child != node:
            queue.append(child)
            while next_sib[child] != child:
                child = next_sib[child]
                queue.append(child)

    return visited


def _enumerate_children_pandas(df: pd.DataFrame, node: int) -> list:
    """Get all children of a node using first_child and next_sibling."""
    first_child = df["first_child_id"].to_list()
    next_sib = df["next_sibling_id"].to_list()

    child = first_child[node]
    if child == node:
        return []

    children = [child]
    while next_sib[child] != child:
        child = next_sib[child]
        children.append(child)

    return children


def _enumerate_children_reverse_pandas(df: pd.DataFrame, node: int) -> list:
    """Get children in reverse using first_child, next_sibling to find last,
    then prev_sibling to walk back."""
    first_child = df["first_child_id"].to_list()
    next_sib = df["next_sibling_id"].to_list()
    prev_sib = df["prev_sibling_id"].to_list()

    child = first_child[node]
    if child == node:
        return []

    # find last child
    last = child
    while next_sib[last] != last:
        last = next_sib[last]

    # walk backward
    children_rev = [last]
    while prev_sib[last] != last:
        last = prev_sib[last]
        children_rev.append(last)

    return children_rev


@pytest.mark.parametrize("mutate", [True, False])
def test_depth_first_traversal_pandas(mutate: bool):
    """Full depth-first traversal of a tree using pandas.

    Tree:
        0 (root)
        +-- 1
        |   +-- 3
        |   |   +-- 7
        |   |   +-- 8
        |   +-- 4
        +-- 2
            +-- 5
            +-- 6
    """
    phylogeny_df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3, 4, 5, 6, 7, 8],
            "ancestor_list": [
                "[None]",
                "[0]",
                "[0]",
                "[1]",
                "[1]",
                "[2]",
                "[2]",
                "[3]",
                "[3]",
            ],
        }
    )

    df = alifestd_mark_first_child_id_asexual(phylogeny_df, mutate=mutate)
    df = alifestd_mark_next_sibling_id_asexual(df, mutate=True)

    visited = _depth_first_pandas(df)
    assert visited == [0, 1, 3, 7, 8, 4, 2, 5, 6]


@pytest.mark.parametrize("mutate", [True, False])
def test_breadth_first_traversal_pandas(mutate: bool):
    """Full breadth-first traversal of a tree using pandas.

    Same tree as above.
    """
    phylogeny_df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3, 4, 5, 6, 7, 8],
            "ancestor_list": [
                "[None]",
                "[0]",
                "[0]",
                "[1]",
                "[1]",
                "[2]",
                "[2]",
                "[3]",
                "[3]",
            ],
        }
    )

    df = alifestd_mark_first_child_id_asexual(phylogeny_df, mutate=mutate)
    df = alifestd_mark_next_sibling_id_asexual(df, mutate=True)

    visited = _breadth_first_pandas(df)
    assert visited == [0, 1, 2, 3, 4, 5, 6, 7, 8]


@pytest.mark.parametrize("mutate", [True, False])
def test_enumerate_children_pandas(mutate: bool):
    """Enumerate children of each node forward and backward.

    Tree:
        0 (root)
        +-- 1
        |   +-- 3
        |   +-- 4
        |   +-- 5
        +-- 2
    """
    phylogeny_df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3, 4, 5],
            "ancestor_list": [
                "[None]",
                "[0]",
                "[0]",
                "[1]",
                "[1]",
                "[1]",
            ],
        }
    )

    df = alifestd_mark_first_child_id_asexual(phylogeny_df, mutate=mutate)
    df = alifestd_mark_next_sibling_id_asexual(df, mutate=True)
    df = alifestd_mark_prev_sibling_id_asexual(df, mutate=True)

    assert _enumerate_children_pandas(df, 0) == [1, 2]
    assert _enumerate_children_pandas(df, 1) == [3, 4, 5]
    assert _enumerate_children_pandas(df, 2) == []  # leaf

    assert _enumerate_children_reverse_pandas(df, 0) == [2, 1]
    assert _enumerate_children_reverse_pandas(df, 1) == [5, 4, 3]
    assert _enumerate_children_reverse_pandas(df, 2) == []  # leaf


@pytest.mark.parametrize(
    "phylogeny_df",
    [
        alifestd_to_working_format(
            pd.read_csv(
                f"{assets_path}/example-standard-toy-asexual-phylogeny.csv"
            )
        ),
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
def test_full_tree_traversal_visits_all_nodes(phylogeny_df: pd.DataFrame):
    """DFS and BFS on real test assets should visit every node exactly once."""
    df_pl = pl.from_pandas(phylogeny_df)

    df_pl = alifestd_mark_first_child_id_polars(df_pl)
    df_pl = alifestd_mark_next_sibling_id_polars(df_pl)

    # convert to pandas for traversal helpers
    df = df_pl.to_pandas()
    df.index = df["id"]

    n = len(df)
    dfs = _depth_first_pandas(df)
    bfs = _breadth_first_pandas(df)

    assert len(dfs) == n
    assert sorted(dfs) == list(range(n))

    assert len(bfs) == n
    assert sorted(bfs) == list(range(n))


@pytest.mark.parametrize(
    "phylogeny_df",
    [
        alifestd_to_working_format(
            pd.read_csv(
                f"{assets_path}/example-standard-toy-asexual-phylogeny.csv"
            )
        ),
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
def test_prev_next_sibling_consistency(phylogeny_df: pd.DataFrame):
    """If A's next_sibling is B, then B's prev_sibling should be A."""
    df_pl = pl.from_pandas(phylogeny_df)

    df_pl = alifestd_mark_next_sibling_id_polars(df_pl)
    df_pl = alifestd_mark_prev_sibling_id_polars(df_pl)

    next_sib = df_pl["next_sibling_id"].to_list()
    prev_sib = df_pl["prev_sibling_id"].to_list()

    for node in range(len(next_sib)):
        ns = next_sib[node]
        if ns != node:  # has a next sibling
            assert prev_sib[ns] == node, (
                f"node {node}'s next_sibling is {ns}, "
                f"but {ns}'s prev_sibling is {prev_sib[ns]}"
            )


@pytest.mark.parametrize(
    "phylogeny_df",
    [
        alifestd_to_working_format(
            pd.read_csv(
                f"{assets_path}/example-standard-toy-asexual-phylogeny.csv"
            )
        ),
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
def test_first_child_is_first_sibling(phylogeny_df: pd.DataFrame):
    """First child should have prev_sibling_id == self (no previous)."""
    df_pl = pl.from_pandas(phylogeny_df)

    df_pl = alifestd_mark_first_child_id_polars(df_pl)
    df_pl = alifestd_mark_prev_sibling_id_polars(df_pl)

    first_child = df_pl["first_child_id"].to_list()
    prev_sib = df_pl["prev_sibling_id"].to_list()

    for node in range(len(first_child)):
        fc = first_child[node]
        if fc != node:  # has children
            assert prev_sib[fc] == fc, (
                f"first child {fc} of node {node} should have "
                f"prev_sibling == self, but got {prev_sib[fc]}"
            )
