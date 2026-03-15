import os

import numpy as np
import polars as pl
import pytest

from phyloframe.legacy import (
    alifestd_unfurl_traversal_preorder_polars,
)

assets_path = os.path.join(os.path.dirname(__file__), "assets")


def _make_contiguous_df(ancestor_ids: np.ndarray) -> pl.DataFrame:
    """Helper to create a contiguous, topologically-sorted polars DataFrame."""
    n = len(ancestor_ids)
    return pl.DataFrame(
        {
            "id": np.arange(n, dtype=np.int64),
            "ancestor_id": np.asarray(ancestor_ids, dtype=np.int64),
        },
    )


def test_empty():
    df = _make_contiguous_df(np.array([], dtype=np.int64))
    result = alifestd_unfurl_traversal_preorder_polars(df)
    assert len(result) == 0


def test_single_node():
    df = _make_contiguous_df(np.array([0]))
    result = alifestd_unfurl_traversal_preorder_polars(df)
    assert result.tolist() == [0]


def test_chain():
    """Linear chain: 0 -> 1 -> 2."""
    df = _make_contiguous_df(np.array([0, 0, 1]))
    result = alifestd_unfurl_traversal_preorder_polars(df)
    assert result.tolist() == [0, 1, 2]


def test_simple_branching():
    """Tree: 0 -> {1, 2}, 1 -> {3}.

    Preorder visits parent first, then children in ascending id order.
    Result: [0, 1, 3, 2]
    """
    df = _make_contiguous_df(np.array([0, 0, 0, 1]))
    result = alifestd_unfurl_traversal_preorder_polars(df)
    assert result.tolist() == [0, 1, 3, 2]


def test_simple4():
    """Tree: 0 -> {1, 2, 4}, 1 -> {3}.

    Preorder: 0, 1, 3, 2, 4
    """
    df = _make_contiguous_df(np.array([0, 0, 0, 1, 0]))
    result = alifestd_unfurl_traversal_preorder_polars(df)
    assert result.tolist() == [0, 1, 3, 2, 4]


def test_multi_root():
    """Two roots: 0 -> {2}, 1 -> {3}."""
    df = _make_contiguous_df(np.array([0, 1, 0, 1]))
    result = alifestd_unfurl_traversal_preorder_polars(df)
    # Root 0 processed first, then root 1
    assert result.tolist() == [0, 2, 1, 3]


def test_star():
    """Star graph: root 0 with children 1, 2, 3, 4."""
    df = _make_contiguous_df(np.array([0, 0, 0, 0, 0]))
    result = alifestd_unfurl_traversal_preorder_polars(df)
    # Children visited in ascending id order
    assert result.tolist() == [0, 1, 2, 3, 4]


def test_deep_tree():
    """Caterpillar: 0 -> 1 -> 2 -> ... -> 9."""
    n = 10
    ancestor_ids = np.arange(n, dtype=np.int64)
    ancestor_ids[0] = 0
    for i in range(1, n):
        ancestor_ids[i] = i - 1

    df = _make_contiguous_df(ancestor_ids)
    result = alifestd_unfurl_traversal_preorder_polars(df)
    assert result.tolist() == list(range(n))


def test_subtree_contiguity():
    """Verify that each subtree's nodes are contiguous in the result.

    Tree: 0 -> {1, 2}, 1 -> {3, 4}, 2 -> {5}.
    """
    df = _make_contiguous_df(np.array([0, 0, 0, 1, 1, 2]))
    result = alifestd_unfurl_traversal_preorder_polars(df)

    result_list = result.tolist()

    # Subtree rooted at 1: {1, 3, 4} should be contiguous
    indices_1 = sorted(result_list.index(x) for x in [1, 3, 4])
    assert indices_1 == list(range(indices_1[0], indices_1[0] + 3))

    # Subtree rooted at 2: {2, 5} should be contiguous
    indices_2 = sorted(result_list.index(x) for x in [2, 5])
    assert indices_2 == list(range(indices_2[0], indices_2[0] + 2))

    # Node 1 should come before all its descendants
    assert result_list.index(1) < result_list.index(3)
    assert result_list.index(1) < result_list.index(4)


def test_valid_preorder():
    """Every node must appear before all its descendants."""
    ancestor_ids = np.array([0, 0, 0, 1, 1, 2])
    df = _make_contiguous_df(ancestor_ids)
    result = alifestd_unfurl_traversal_preorder_polars(df)
    result_list = result.tolist()
    pos = {node: i for i, node in enumerate(result_list)}

    for child, parent in enumerate(ancestor_ids):
        if child != parent:
            assert pos[parent] < pos[child], (
                f"parent {parent} at pos {pos[parent]} should come before "
                f"child {child} at pos {pos[child]}"
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
    phylogeny_df = pl.read_csv(f"{assets_path}/{phylogeny_csv}")
    if "ancestor_list" in phylogeny_df.columns:
        phylogeny_df = phylogeny_df.with_columns(
            ancestor_id=pl.col("ancestor_list")
            .str.extract(r"(\d+)", 1)
            .cast(pl.Int64, strict=False)
            .fill_null(pl.col("id")),
        )

    # Sort by id to ensure topological order, then remap to contiguous
    phylogeny_df = phylogeny_df.sort("id")
    ids = phylogeny_df["id"].to_numpy()
    if not (ids == np.arange(len(ids))).all():
        id_map = {int(old): new for new, old in enumerate(ids)}
        ancestor_ids = np.array(
            [id_map[int(a)] for a in phylogeny_df["ancestor_id"].to_numpy()],
            dtype=np.int64,
        )
        phylogeny_df = _make_contiguous_df(ancestor_ids)

    result = alifestd_unfurl_traversal_preorder_polars(phylogeny_df)

    ancestor_ids = phylogeny_df["ancestor_id"].to_numpy()
    n = len(ancestor_ids)
    assert len(result) == n
    assert set(result) == set(range(n))

    # Verify valid preorder: every parent before its children
    pos = {node: i for i, node in enumerate(result)}
    for child in range(n):
        parent = ancestor_ids[child]
        if child != parent:
            assert pos[parent] < pos[child]

    # Verify subtree contiguity
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
        all_nodes = desc | {node}
        positions = sorted(pos[d] for d in all_nodes)
        assert positions == list(
            range(positions[0], positions[0] + len(positions))
        ), f"subtree of node {node} is not contiguous"


def test_non_contiguous_ids():
    """Test that non-contiguous IDs raise NotImplementedError."""
    df = pl.DataFrame(
        {
            "id": [10, 20, 30],
            "ancestor_id": [10, 10, 20],
        },
    )
    with pytest.raises(NotImplementedError):
        alifestd_unfurl_traversal_preorder_polars(df)


def test_non_topologically_sorted():
    """Test that non-topologically-sorted data raises NotImplementedError."""
    df = pl.DataFrame(
        {
            "id": [0, 1, 2],
            "ancestor_id": [1, 1, 0],
        },
    )
    with pytest.raises(NotImplementedError):
        alifestd_unfurl_traversal_preorder_polars(df)


def test_with_num_children_col():
    """Test that pre-existing num_children column is reused."""
    ancestor_ids = np.array([0, 0, 0, 1], dtype=np.int64)
    df = _make_contiguous_df(ancestor_ids)
    df = df.with_columns(num_children=pl.Series([2, 1, 0, 0]))
    result = alifestd_unfurl_traversal_preorder_polars(df)
    assert result.tolist() == [0, 1, 3, 2]


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_lazyframe(apply):
    """Test that LazyFrame input works."""
    df = apply(_make_contiguous_df(np.array([0, 0, 1])))
    result = alifestd_unfurl_traversal_preorder_polars(df)
    assert result.tolist() == [0, 1, 2]


def test_with_ancestor_list_col():
    """Test with ancestor_list instead of ancestor_id."""
    df = pl.DataFrame(
        {
            "id": [0, 1, 2],
            "ancestor_list": ["[None]", "[0]", "[1]"],
        },
    )
    result = alifestd_unfurl_traversal_preorder_polars(df)
    assert result.tolist() == [0, 1, 2]
