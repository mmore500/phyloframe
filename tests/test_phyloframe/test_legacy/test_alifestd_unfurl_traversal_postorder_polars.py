import os

import numpy as np
import polars as pl
import pytest

from phyloframe.legacy import (
    alifestd_mark_csr_children_polars,
    alifestd_mark_csr_offsets_polars,
    alifestd_mark_first_child_id_polars,
    alifestd_mark_next_sibling_id_polars,
    alifestd_mark_num_children_polars,
    alifestd_unfurl_traversal_postorder_polars,
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
    result = alifestd_unfurl_traversal_postorder_polars(df)
    assert len(result) == 0


def test_single_node():
    df = _make_contiguous_df(np.array([0]))
    result = alifestd_unfurl_traversal_postorder_polars(df)
    assert result.tolist() == [0]


def test_chain():
    """Linear chain: 0 -> 1 -> 2."""
    df = _make_contiguous_df(np.array([0, 0, 1]))
    result = alifestd_unfurl_traversal_postorder_polars(df)
    assert result.tolist() == [2, 1, 0]


def _assert_valid_postorder(result, ancestor_ids):
    """Assert that result is a valid postorder: parents after children."""
    result_list = (
        result.tolist() if hasattr(result, "tolist") else list(result)
    )
    pos = {node: i for i, node in enumerate(result_list)}
    for child, parent in enumerate(ancestor_ids):
        if child != parent:
            assert pos[parent] > pos[child], (
                f"parent {parent} at pos {pos[parent]} should come after "
                f"child {child} at pos {pos[child]}"
            )


def test_simple_branching():
    """Tree: 0 -> {1, 2}, 1 -> {3}."""
    ancestor_ids = np.array([0, 0, 0, 1])
    df = _make_contiguous_df(ancestor_ids)
    result = alifestd_unfurl_traversal_postorder_polars(df)
    assert set(result) == set(range(len(ancestor_ids)))
    _assert_valid_postorder(result, ancestor_ids)
    # Root must be last
    assert result[-1] == 0


def test_simple4():
    """Tree: 0 -> {1, 2, 4}, 1 -> {3}."""
    ancestor_ids = np.array([0, 0, 0, 1, 0])
    df = _make_contiguous_df(ancestor_ids)
    result = alifestd_unfurl_traversal_postorder_polars(df)
    assert set(result) == set(range(len(ancestor_ids)))
    _assert_valid_postorder(result, ancestor_ids)
    assert result[-1] == 0


def test_star():
    """Star graph: root 0 with children 1, 2, 3, 4."""
    ancestor_ids = np.array([0, 0, 0, 0, 0])
    df = _make_contiguous_df(ancestor_ids)
    result = alifestd_unfurl_traversal_postorder_polars(df)
    assert set(result) == set(range(len(ancestor_ids)))
    _assert_valid_postorder(result, ancestor_ids)
    assert result[-1] == 0


def test_deep_tree():
    """Caterpillar: 0 -> 1 -> 2 -> ... -> 9."""
    n = 10
    ancestor_ids = np.arange(n, dtype=np.int64)
    ancestor_ids[0] = 0
    for i in range(1, n):
        ancestor_ids[i] = i - 1

    df = _make_contiguous_df(ancestor_ids)
    result = alifestd_unfurl_traversal_postorder_polars(df)
    assert result.tolist() == list(range(n - 1, -1, -1))


def test_valid_postorder():
    """Every node must appear after all its descendants."""
    ancestor_ids = np.array([0, 0, 0, 1, 1, 2])
    df = _make_contiguous_df(ancestor_ids)
    result = alifestd_unfurl_traversal_postorder_polars(df)
    result_list = result.tolist()
    pos = {node: i for i, node in enumerate(result_list)}

    for child, parent in enumerate(ancestor_ids):
        if child != parent:
            assert pos[parent] > pos[child], (
                f"parent {parent} at pos {pos[parent]} should come after "
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

    result = alifestd_unfurl_traversal_postorder_polars(phylogeny_df)

    ancestor_ids = phylogeny_df["ancestor_id"].to_numpy()
    n = len(ancestor_ids)
    assert len(result) == n
    assert set(result) == set(range(n))

    # Verify valid postorder: every parent after its children
    pos = {node: i for i, node in enumerate(result)}
    for child in range(n):
        parent = ancestor_ids[child]
        if child != parent:
            assert pos[parent] > pos[child]


def test_non_contiguous_ids():
    """Test that non-contiguous IDs raise NotImplementedError."""
    df = pl.DataFrame(
        {
            "id": [10, 20, 30],
            "ancestor_id": [10, 10, 20],
        },
    )
    with pytest.raises(NotImplementedError):
        alifestd_unfurl_traversal_postorder_polars(df)


def test_non_topologically_sorted():
    """Test that non-topologically-sorted data raises NotImplementedError."""
    df = pl.DataFrame(
        {
            "id": [0, 1, 2],
            "ancestor_id": [1, 1, 0],
        },
    )
    with pytest.raises(NotImplementedError):
        alifestd_unfurl_traversal_postorder_polars(df)


def test_with_node_depth_col():
    """Test that pre-existing node_depth column is reused."""
    ancestor_ids = np.array([0, 0, 0, 1], dtype=np.int64)
    df = _make_contiguous_df(ancestor_ids)
    df = df.with_columns(node_depth=pl.Series([0, 1, 1, 2]))
    result = alifestd_unfurl_traversal_postorder_polars(df)
    assert set(result) == set(range(len(ancestor_ids)))
    _assert_valid_postorder(result, ancestor_ids)


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
    result = alifestd_unfurl_traversal_postorder_polars(df)
    assert result.tolist() == [2, 1, 0]


def test_with_ancestor_list_col():
    """Test with ancestor_list instead of ancestor_id."""
    df = pl.DataFrame(
        {
            "id": [0, 1, 2],
            "ancestor_list": ["[None]", "[0]", "[1]"],
        },
    )
    result = alifestd_unfurl_traversal_postorder_polars(df)
    assert result.tolist() == [2, 1, 0]


def test_matches_pandas():
    """Verify polars result is a valid postorder like pandas result."""
    import pandas as pd

    from phyloframe.legacy import alifestd_unfurl_traversal_postorder_asexual

    ancestor_ids = np.array([0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    n = len(ancestor_ids)

    pd_df = pd.DataFrame(
        {
            "id": np.arange(n),
            "ancestor_id": ancestor_ids,
        },
    )
    pl_df = _make_contiguous_df(ancestor_ids)

    pd_result = alifestd_unfurl_traversal_postorder_asexual(pd_df, mutate=True)
    pl_result = alifestd_unfurl_traversal_postorder_polars(pl_df)

    # Both should be valid permutations and postorders
    assert set(pd_result) == set(pl_result) == set(range(n))
    _assert_valid_postorder(pd_result, ancestor_ids)
    _assert_valid_postorder(pl_result, ancestor_ids)


def _add_sibling_cols(df: pl.DataFrame) -> pl.DataFrame:
    """Add first_child_id and next_sibling_id columns."""
    df = alifestd_mark_first_child_id_polars(df)
    df = alifestd_mark_next_sibling_id_polars(df)
    return df


def _add_csr_cols(df: pl.DataFrame) -> pl.DataFrame:
    """Add num_children, csr_offsets, and csr_children columns."""
    df = alifestd_mark_num_children_polars(df)
    df = alifestd_mark_csr_offsets_polars(df)
    df = alifestd_mark_csr_children_polars(df)
    return df


def test_with_sibling_cols():
    """Test that sibling columns trigger the sibling JIT path."""
    ancestor_ids = np.array([0, 0, 0, 1], dtype=np.int64)
    df = _add_sibling_cols(_make_contiguous_df(ancestor_ids))
    result = alifestd_unfurl_traversal_postorder_polars(df)
    assert set(result) == set(range(len(ancestor_ids)))
    _assert_valid_postorder(result, ancestor_ids)
    assert result[-1] == 0


def test_with_csr_cols():
    """Test that CSR columns trigger the CSR JIT path."""
    ancestor_ids = np.array([0, 0, 0, 1], dtype=np.int64)
    df = _add_csr_cols(_make_contiguous_df(ancestor_ids))
    result = alifestd_unfurl_traversal_postorder_polars(df)
    assert set(result) == set(range(len(ancestor_ids)))
    _assert_valid_postorder(result, ancestor_ids)
    assert result[-1] == 0


@pytest.mark.parametrize(
    "phylogeny_csv",
    [
        "example-standard-toy-asexual-phylogeny.csv",
        "nk_ecoeaselection.csv",
        "nk_lexicaseselection.csv",
        "nk_tournamentselection.csv",
    ],
)
@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(_add_sibling_cols, id="sibling"),
        pytest.param(_add_csr_cols, id="csr"),
    ],
)
def test_fuzz_jit_paths(phylogeny_csv: str, apply):
    """Fuzz test sibling-based and CSR-based JIT paths."""
    phylogeny_df = pl.read_csv(f"{assets_path}/{phylogeny_csv}")
    if "ancestor_list" in phylogeny_df.columns:
        phylogeny_df = phylogeny_df.with_columns(
            ancestor_id=pl.col("ancestor_list")
            .str.extract(r"(\d+)", 1)
            .cast(pl.Int64, strict=False)
            .fill_null(pl.col("id")),
        )

    phylogeny_df = phylogeny_df.sort("id")
    ids = phylogeny_df["id"].to_numpy()
    if not (ids == np.arange(len(ids))).all():
        id_map = {int(old): new for new, old in enumerate(ids)}
        ancestor_ids = np.array(
            [id_map[int(a)] for a in phylogeny_df["ancestor_id"].to_numpy()],
            dtype=np.int64,
        )
        phylogeny_df = _make_contiguous_df(ancestor_ids)

    phylogeny_df = apply(phylogeny_df)
    result = alifestd_unfurl_traversal_postorder_polars(phylogeny_df)

    ancestor_ids = phylogeny_df["ancestor_id"].to_numpy()
    n = len(ancestor_ids)
    assert len(result) == n
    assert set(result) == set(range(n))

    pos = {node: i for i, node in enumerate(result)}
    for child in range(n):
        parent = ancestor_ids[child]
        if child != parent:
            assert pos[parent] > pos[child]
