import os
import typing

import numpy as np
import pandas as pd
import pytest

from phyloframe.legacy import (
    alifestd_mark_csr_children_asexual,
    alifestd_mark_csr_offsets_asexual,
    alifestd_mark_first_child_id_asexual,
    alifestd_mark_next_sibling_id_asexual,
    alifestd_mark_num_children_asexual,
    alifestd_unfurl_traversal_postorder_contiguous_asexual,
)

assets_path = os.path.join(os.path.dirname(__file__), "assets")


def _make_contiguous_df(ancestor_ids) -> pd.DataFrame:
    """Helper to create a contiguous, topologically-sorted pandas DataFrame."""
    n = len(ancestor_ids)
    return pd.DataFrame(
        {
            "id": np.arange(n, dtype=np.int64),
            "ancestor_id": np.asarray(ancestor_ids, dtype=np.int64),
        },
    )


def _prep_noop(df):
    return df


def _prep_csr(df):
    df = alifestd_mark_num_children_asexual(df, mutate=True)
    df = alifestd_mark_csr_offsets_asexual(df, mutate=True)
    df = alifestd_mark_csr_children_asexual(df, mutate=True)
    return df


def _prep_linked_list(df):
    df = alifestd_mark_first_child_id_asexual(df, mutate=True)
    df = alifestd_mark_next_sibling_id_asexual(df, mutate=True)
    return df


_prep_params = pytest.mark.parametrize(
    "apply",
    [
        pytest.param(_prep_noop, id="no-precompute"),
        pytest.param(_prep_csr, id="csr"),
        pytest.param(_prep_linked_list, id="linked-list"),
    ],
)


@_prep_params
def test_empty(apply: typing.Callable):
    df = _make_contiguous_df(np.array([], dtype=np.int64))
    result = alifestd_unfurl_traversal_postorder_contiguous_asexual(apply(df))
    assert len(result) == 0


@_prep_params
def test_single_node(apply: typing.Callable):
    df = _make_contiguous_df(np.array([0]))
    result = alifestd_unfurl_traversal_postorder_contiguous_asexual(apply(df))
    assert result.tolist() == [0]


@_prep_params
def test_chain(apply: typing.Callable):
    """Linear chain: 0 -> 1 -> 2."""
    df = _make_contiguous_df(np.array([0, 0, 1]))
    result = alifestd_unfurl_traversal_postorder_contiguous_asexual(apply(df))
    assert result.tolist() == [2, 1, 0]


@_prep_params
def test_simple_branching(apply: typing.Callable):
    """Tree: 0 -> {1, 2}, 1 -> {3}.

    DFS visits children in ascending id order (highest-id on top of stack,
    processed first). So children of 0 are visited 2 first, then subtree
    of 1 (which includes 3).

    Result: [2, 3, 1, 0]
    """
    df = _make_contiguous_df(np.array([0, 0, 0, 1]))
    result = alifestd_unfurl_traversal_postorder_contiguous_asexual(apply(df))
    assert result.tolist() == [2, 3, 1, 0]


@_prep_params
def test_simple4(apply: typing.Callable):
    """Tree: 0 -> {1, 2, 4}, 1 -> {3}.

    Children of 0 visited as: 4, 2, then subtree of 1 (3, 1).
    Result: [4, 2, 3, 1, 0]
    """
    df = _make_contiguous_df(np.array([0, 0, 0, 1, 0]))
    result = alifestd_unfurl_traversal_postorder_contiguous_asexual(apply(df))
    assert result.tolist() == [4, 2, 3, 1, 0]


@_prep_params
def test_multi_root(apply: typing.Callable):
    """Two roots: 0 -> {2}, 1 -> {3}."""
    df = _make_contiguous_df(np.array([0, 1, 0, 1]))
    result = alifestd_unfurl_traversal_postorder_contiguous_asexual(apply(df))
    # Root 0 processed first, then root 1
    assert result.tolist() == [2, 0, 3, 1]


@_prep_params
def test_star(apply: typing.Callable):
    """Star graph: root 0 with children 1, 2, 3, 4."""
    df = _make_contiguous_df(np.array([0, 0, 0, 0, 0]))
    result = alifestd_unfurl_traversal_postorder_contiguous_asexual(apply(df))
    # Highest-id child on top of stack, processed first
    assert result.tolist() == [4, 3, 2, 1, 0]


@_prep_params
def test_deep_tree(apply: typing.Callable):
    """Caterpillar: 0 -> 1 -> 2 -> ... -> 9."""
    n = 10
    ancestor_ids = np.arange(n, dtype=np.int64)
    ancestor_ids[0] = 0
    for i in range(1, n):
        ancestor_ids[i] = i - 1

    df = _make_contiguous_df(ancestor_ids)
    result = alifestd_unfurl_traversal_postorder_contiguous_asexual(apply(df))
    assert result.tolist() == list(range(n - 1, -1, -1))


@_prep_params
def test_subtree_contiguity(apply: typing.Callable):
    """Verify that each subtree's nodes are contiguous in the result.

    Tree: 0 -> {1, 2}, 1 -> {3, 4}, 2 -> {5}.
    """
    df = _make_contiguous_df(np.array([0, 0, 0, 1, 1, 2]))
    result = alifestd_unfurl_traversal_postorder_contiguous_asexual(apply(df))

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


@_prep_params
def test_valid_postorder(apply: typing.Callable):
    """Every node must appear after all its descendants."""
    ancestor_ids = np.array([0, 0, 0, 1, 1, 2])
    df = _make_contiguous_df(ancestor_ids)
    result = alifestd_unfurl_traversal_postorder_contiguous_asexual(apply(df))
    result_list = result.tolist()
    pos = {node: i for i, node in enumerate(result_list)}

    for child, parent in enumerate(ancestor_ids):
        if child != parent:
            assert pos[child] < pos[parent], (
                f"child {child} at pos {pos[child]} should come before "
                f"parent {parent} at pos {pos[parent]}"
            )


@pytest.mark.parametrize("mutate", [True, False])
def test_mutate(mutate: bool):
    df = _make_contiguous_df(np.array([0, 0, 1]))
    original_df = df.copy()
    result = alifestd_unfurl_traversal_postorder_contiguous_asexual(
        df,
        mutate=mutate,
    )
    assert result.tolist() == [2, 1, 0]
    if not mutate:
        assert original_df.equals(df)


@pytest.mark.parametrize(
    "phylogeny_csv",
    [
        "example-standard-toy-asexual-phylogeny.csv",
        "nk_ecoeaselection.csv",
        "nk_lexicaseselection.csv",
        "nk_tournamentselection.csv",
    ],
)
@_prep_params
def test_fuzz(phylogeny_csv: str, apply: typing.Callable):
    phylogeny_df = pd.read_csv(f"{assets_path}/{phylogeny_csv}")

    # Sort by id and remap to contiguous 0..n-1
    phylogeny_df = phylogeny_df.sort_values("id").reset_index(drop=True)
    from phyloframe.legacy import alifestd_try_add_ancestor_id_col

    phylogeny_df = alifestd_try_add_ancestor_id_col(phylogeny_df.copy())
    ids = phylogeny_df["id"].to_numpy()
    id_map = {int(old): new for new, old in enumerate(ids)}
    ancestor_ids = np.array(
        [id_map[int(a)] for a in phylogeny_df["ancestor_id"].to_numpy()],
        dtype=np.int64,
    )
    df = _make_contiguous_df(ancestor_ids)

    result = alifestd_unfurl_traversal_postorder_contiguous_asexual(apply(df))

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


def test_non_contiguous_ids():
    """Test that non-contiguous IDs raise NotImplementedError."""
    df = pd.DataFrame(
        {
            "id": [10, 20, 30],
            "ancestor_id": [10, 10, 20],
        },
    )
    with pytest.raises(NotImplementedError):
        alifestd_unfurl_traversal_postorder_contiguous_asexual(df)


def test_non_topologically_sorted():
    """Test that non-topologically-sorted data raises NotImplementedError."""
    df = pd.DataFrame(
        {
            "id": [0, 1, 2],
            "ancestor_id": [1, 1, 0],
        },
    )
    with pytest.raises(NotImplementedError):
        alifestd_unfurl_traversal_postorder_contiguous_asexual(df)


def test_with_num_children_col():
    """Test that pre-existing num_children column is reused."""
    ancestor_ids = np.array([0, 0, 0, 1], dtype=np.int64)
    df = _make_contiguous_df(ancestor_ids)
    df["num_children"] = [2, 1, 0, 0]
    result = alifestd_unfurl_traversal_postorder_contiguous_asexual(df)
    assert result.tolist() == [2, 3, 1, 0]


def test_with_ancestor_list_col():
    """Test with ancestor_list instead of ancestor_id."""
    df = pd.DataFrame(
        {
            "id": [0, 1, 2],
            "ancestor_list": ["[None]", "[0]", "[1]"],
        },
    )
    result = alifestd_unfurl_traversal_postorder_contiguous_asexual(df)
    assert result.tolist() == [2, 1, 0]


def test_with_sibling_cols():
    """Test sibling fast path with first_child_id + next_sibling_id."""
    df = _make_contiguous_df(np.array([0, 0, 0, 1]))
    df["first_child_id"] = [1, 3, 2, 3]
    df["next_sibling_id"] = [0, 2, 2, 3]
    result = alifestd_unfurl_traversal_postorder_contiguous_asexual(df)
    assert result.tolist() == [2, 3, 1, 0]


@pytest.mark.parametrize(
    "phylogeny_csv",
    [
        "example-standard-toy-asexual-phylogeny.csv",
        "nk_ecoeaselection.csv",
        "nk_lexicaseselection.csv",
        "nk_tournamentselection.csv",
    ],
)
def test_sibling_fast_path_matches_baseline(phylogeny_csv: str):
    """Verify sibling-based fast path gives same result as CSR-based."""
    from phyloframe.legacy import (
        alifestd_mark_first_child_id_asexual,
        alifestd_mark_next_sibling_id_asexual,
        alifestd_try_add_ancestor_id_col,
    )

    phylogeny_df = pd.read_csv(f"{assets_path}/{phylogeny_csv}")
    phylogeny_df = phylogeny_df.sort_values("id").reset_index(drop=True)
    phylogeny_df = alifestd_try_add_ancestor_id_col(phylogeny_df.copy())
    ids = phylogeny_df["id"].to_numpy()
    id_map = {int(old): new for new, old in enumerate(ids)}
    ancestor_ids = np.array(
        [id_map[int(a)] for a in phylogeny_df["ancestor_id"].to_numpy()],
        dtype=np.int64,
    )
    df_base = _make_contiguous_df(ancestor_ids)

    result_base = alifestd_unfurl_traversal_postorder_contiguous_asexual(
        df_base,
    )

    df_sib = df_base.copy()
    df_sib = alifestd_mark_first_child_id_asexual(df_sib, mutate=True)
    df_sib = alifestd_mark_next_sibling_id_asexual(df_sib, mutate=True)
    result_sib = alifestd_unfurl_traversal_postorder_contiguous_asexual(
        df_sib,
    )

    assert result_base.tolist() == result_sib.tolist()
