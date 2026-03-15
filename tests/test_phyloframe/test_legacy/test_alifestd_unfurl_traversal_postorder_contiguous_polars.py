import os

import numpy as np
import pandas as pd
import pytest

from phyloframe.legacy import (
    alifestd_has_contiguous_ids,
    alifestd_try_add_ancestor_id_col,
)
from phyloframe.legacy._alifestd_unfurl_traversal_postorder_contiguous_polars import (
    _alifestd_unfurl_traversal_postorder_contiguous_polars_jit,
)

assets_path = os.path.join(os.path.dirname(__file__), "assets")


def _get_contiguous_ancestor_ids(phylogeny_df: pd.DataFrame) -> np.ndarray:
    """Helper to get ancestor_ids, remapping to contiguous 0..n-1."""
    phylogeny_df = alifestd_try_add_ancestor_id_col(phylogeny_df.copy())
    if alifestd_has_contiguous_ids(phylogeny_df):
        return phylogeny_df["ancestor_id"].to_numpy()
    # Remap to contiguous IDs
    id_map = {old: new for new, old in enumerate(phylogeny_df["id"])}
    return np.array(
        [id_map[a] for a in phylogeny_df["ancestor_id"]], dtype=np.int64
    )


def test_empty():
    ancestor_ids = np.array([], dtype=np.int64)
    result = _alifestd_unfurl_traversal_postorder_contiguous_polars_jit(
        ancestor_ids,
    )
    assert len(result) == 0


def test_single_node():
    ancestor_ids = np.array([0])
    result = _alifestd_unfurl_traversal_postorder_contiguous_polars_jit(
        ancestor_ids,
    )
    assert result.tolist() == [0]


def test_chain():
    """Linear chain: 0 -> 1 -> 2."""
    ancestor_ids = np.array([0, 0, 1])
    result = _alifestd_unfurl_traversal_postorder_contiguous_polars_jit(
        ancestor_ids,
    )
    assert result.tolist() == [2, 1, 0]


def test_simple_branching():
    """Tree: 0 -> {1, 2}, 1 -> {3}.

    DFS visits children in ascending id order (highest-id on top of stack,
    processed first). So children of 0 are visited 2 first, then subtree
    of 1 (which includes 3).

    Result: [2, 3, 1, 0]
    """
    ancestor_ids = np.array([0, 0, 0, 1])
    result = _alifestd_unfurl_traversal_postorder_contiguous_polars_jit(
        ancestor_ids,
    )
    assert result.tolist() == [2, 3, 1, 0]


def test_simple4():
    """Tree: 0 -> {1, 2, 4}, 1 -> {3}.

    Children of 0 visited as: 4, 2, then subtree of 1 (3, 1).
    Result: [4, 2, 3, 1, 0]
    """
    ancestor_ids = np.array([0, 0, 0, 1, 0])
    result = _alifestd_unfurl_traversal_postorder_contiguous_polars_jit(
        ancestor_ids,
    )
    assert result.tolist() == [4, 2, 3, 1, 0]


def test_multi_root():
    """Two roots: 0 -> {2}, 1 -> {3}."""
    ancestor_ids = np.array([0, 1, 0, 1])
    result = _alifestd_unfurl_traversal_postorder_contiguous_polars_jit(
        ancestor_ids,
    )
    # Root 0 processed first, then root 1
    assert result.tolist() == [2, 0, 3, 1]


def test_star():
    """Star graph: root 0 with children 1, 2, 3, 4."""
    ancestor_ids = np.array([0, 0, 0, 0, 0])
    result = _alifestd_unfurl_traversal_postorder_contiguous_polars_jit(
        ancestor_ids,
    )
    # Highest-id child on top of stack, processed first
    assert result.tolist() == [4, 3, 2, 1, 0]


def test_deep_tree():
    """Caterpillar: 0 -> 1 -> 2 -> ... -> 9."""
    n = 10
    ancestor_ids = np.arange(n, dtype=np.int64)
    ancestor_ids[0] = 0
    for i in range(1, n):
        ancestor_ids[i] = i - 1

    result = _alifestd_unfurl_traversal_postorder_contiguous_polars_jit(
        ancestor_ids,
    )
    assert result.tolist() == list(range(n - 1, -1, -1))


def test_subtree_contiguity():
    """Verify that each subtree's nodes are contiguous in the result.

    Tree: 0 -> {1, 2}, 1 -> {3, 4}, 2 -> {5}.
    """
    ancestor_ids = np.array([0, 0, 0, 1, 1, 2])
    result = _alifestd_unfurl_traversal_postorder_contiguous_polars_jit(
        ancestor_ids,
    )

    result_list = result.tolist()

    # Subtree rooted at 1: {1, 3, 4} should be contiguous
    indices_1 = sorted(result_list.index(x) for x in [1, 3, 4])
    assert indices_1 == list(range(indices_1[0], indices_1[0] + 3))

    # Subtree rooted at 2: {2, 5} should be contiguous
    indices_2 = sorted(result_list.index(x) for x in [2, 5])
    assert indices_2 == list(range(indices_2[0], indices_2[0] + 2))

    # Node 1 should come after all its descendants
    assert result_list.index(1) > result_list.index(3)
    assert result_list.index(1) > result_list.index(4)


def test_valid_postorder():
    """Every node must appear after all its descendants."""
    ancestor_ids = np.array([0, 0, 0, 1, 1, 2])
    result = _alifestd_unfurl_traversal_postorder_contiguous_polars_jit(
        ancestor_ids,
    )
    result_list = result.tolist()
    pos = {node: i for i, node in enumerate(result_list)}

    for child, parent in enumerate(ancestor_ids):
        if child != parent:
            assert pos[child] < pos[parent], (
                f"child {child} at pos {pos[child]} should come before "
                f"parent {parent} at pos {pos[parent]}"
            )


@pytest.mark.parametrize(
    "phylogeny_csv",
    [
        "example-standard-toy-asexual-phylogeny.csv",
        "nk_ecoeaselection.csv",
        "nk_lexicaseselection.csv",
        "nk_tournamentselection.csv",
    ],
)
def test_fuzz(phylogeny_csv: str):
    phylogeny_df = pd.read_csv(f"{assets_path}/{phylogeny_csv}")
    ancestor_ids = _get_contiguous_ancestor_ids(phylogeny_df)

    result = _alifestd_unfurl_traversal_postorder_contiguous_polars_jit(
        ancestor_ids,
    )

    n = len(ancestor_ids)
    assert len(result) == n
    assert set(result) == set(range(n))

    # Verify valid postorder: every child before its parent
    pos = {node: i for i, node in enumerate(result)}
    for child in range(n):
        parent = ancestor_ids[child]
        if child != parent:
            assert pos[child] < pos[parent]

    # Verify subtree contiguity: for each internal node, all descendants
    # must form a contiguous block in the result
    children_of = [[] for _ in range(n)]
    for i in range(n):
        if ancestor_ids[i] != i:
            children_of[ancestor_ids[i]].append(i)

    def get_descendants(node):
        desc = set()
        stack = [node]
        while stack:
            cur = stack.pop()
            for c in children_of[cur]:
                desc.add(c)
                stack.append(c)
        return desc

    for node in range(n):
        desc = get_descendants(node)
        if not desc:
            continue
        positions = sorted(pos[d] for d in desc)
        assert positions == list(
            range(positions[0], positions[0] + len(positions))
        ), f"subtree of node {node} is not contiguous"
