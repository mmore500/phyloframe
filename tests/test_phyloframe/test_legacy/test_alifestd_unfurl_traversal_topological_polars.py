import os

import numpy as np
import polars as pl
import pytest

from phyloframe.legacy import (
    alifestd_unfurl_traversal_topological_polars,
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
    result = alifestd_unfurl_traversal_topological_polars(df)
    assert len(result) == 0


def test_single_node():
    df = _make_contiguous_df(np.array([0]))
    result = alifestd_unfurl_traversal_topological_polars(df)
    assert result.tolist() == [0]


def test_chain():
    """Linear chain: 0 -> 1 -> 2."""
    df = _make_contiguous_df(np.array([0, 0, 1]))
    result = alifestd_unfurl_traversal_topological_polars(df)
    assert result.tolist() == [0, 1, 2]


def test_simple_branching():
    """Tree: 0 -> {1, 2}, 1 -> {3}.

    Already sorted, so return [0, 1, 2, 3].
    """
    df = _make_contiguous_df(np.array([0, 0, 0, 1]))
    result = alifestd_unfurl_traversal_topological_polars(df)
    assert result.tolist() == [0, 1, 2, 3]


def test_multi_root():
    """Two roots: 0 -> {2}, 1 -> {3}.

    Already sorted, so return [0, 1, 2, 3].
    """
    df = _make_contiguous_df(np.array([0, 1, 0, 1]))
    result = alifestd_unfurl_traversal_topological_polars(df)
    assert result.tolist() == [0, 1, 2, 3]


def test_star():
    """Star graph: root 0 with children 1, 2, 3, 4."""
    df = _make_contiguous_df(np.array([0, 0, 0, 0, 0]))
    result = alifestd_unfurl_traversal_topological_polars(df)
    assert result.tolist() == [0, 1, 2, 3, 4]


def test_deep_tree():
    """Caterpillar: 0 -> 1 -> 2 -> ... -> 9."""
    n = 10
    ancestor_ids = np.arange(n, dtype=np.int64)
    ancestor_ids[0] = 0
    for i in range(1, n):
        ancestor_ids[i] = i - 1

    df = _make_contiguous_df(ancestor_ids)
    result = alifestd_unfurl_traversal_topological_polars(df)
    assert result.tolist() == list(range(n))


def test_unsorted_computes_order():
    """Test that unsorted data produces valid topological index order."""
    # ancestor_ids where node 0 has parent 1, so not topologically sorted
    df = pl.DataFrame(
        {
            "id": [0, 1, 2],
            "ancestor_id": [1, 1, 0],
        },
    )
    result = alifestd_unfurl_traversal_topological_polars(df)
    result_list = result.tolist()
    # Result contains row indices; verify parent rows come before children
    ancestor_ids = df["ancestor_id"].to_numpy()
    pos = {idx: i for i, idx in enumerate(result_list)}
    for child_idx in range(len(ancestor_ids)):
        parent_idx = ancestor_ids[child_idx]
        if parent_idx != child_idx:
            assert pos[parent_idx] < pos[child_idx]


def test_valid_topological_order():
    """Every parent must appear before its children."""
    ancestor_ids = np.array([0, 0, 0, 1, 1, 2])
    df = _make_contiguous_df(ancestor_ids)
    result = alifestd_unfurl_traversal_topological_polars(df)
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

    result = alifestd_unfurl_traversal_topological_polars(phylogeny_df)

    ancestor_ids = phylogeny_df["ancestor_id"].to_numpy()
    n = len(ancestor_ids)
    assert len(result) == n
    assert set(result) == set(range(n))

    # Verify valid topological order: every parent before its children
    pos = {node: i for i, node in enumerate(result)}
    for child in range(n):
        parent = ancestor_ids[child]
        if child != parent:
            assert pos[parent] < pos[child]


def test_non_contiguous_ids():
    """Test that non-contiguous IDs raise NotImplementedError."""
    df = pl.DataFrame(
        {
            "id": [10, 20, 30],
            "ancestor_id": [10, 10, 20],
        },
    )
    with pytest.raises(NotImplementedError):
        alifestd_unfurl_traversal_topological_polars(df)


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
    result = alifestd_unfurl_traversal_topological_polars(df)
    assert result.tolist() == [0, 1, 2]


def test_with_ancestor_list_col():
    """Test with ancestor_list instead of ancestor_id."""
    df = pl.DataFrame(
        {
            "id": [0, 1, 2],
            "ancestor_list": ["[None]", "[0]", "[1]"],
        },
    )
    result = alifestd_unfurl_traversal_topological_polars(df)
    assert result.tolist() == [0, 1, 2]
