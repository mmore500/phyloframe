import os

import numpy as np
import polars as pl
import pytest

from phyloframe.legacy import (
    alifestd_unfurl_traversal_inorder_polars,
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
    result = alifestd_unfurl_traversal_inorder_polars(df)
    assert len(result) == 0


def test_simple_balanced():
    """Tree structure:
            0
          /   \\
         1     2
        / \\   / \\
       3   4 5   6

    Inorder: 3, 1, 4, 0, 5, 2, 6
    """
    df = _make_contiguous_df(np.array([0, 0, 0, 1, 1, 2, 2]))
    result = alifestd_unfurl_traversal_inorder_polars(df)
    assert tuple(result.tolist()) == (3, 1, 4, 0, 5, 2, 6)


def test_left_heavy():
    """Tree structure:
            0
          /   \\
         1     2
        / \\
       3   4
          / \\
         5   6

    Inorder: 3, 1, 5, 4, 6, 0, 2
    """
    df = _make_contiguous_df(np.array([0, 0, 0, 1, 1, 4, 4]))
    result = alifestd_unfurl_traversal_inorder_polars(df)
    assert tuple(result.tolist()) == (3, 1, 5, 4, 6, 0, 2)


def test_permutation():
    """Verify result is a valid permutation of node ids."""
    ancestor_ids = np.array([0, 0, 0, 1, 1, 2, 2], dtype=np.int64)
    df = _make_contiguous_df(ancestor_ids)
    result = alifestd_unfurl_traversal_inorder_polars(df)
    assert len(result) == len(ancestor_ids)
    assert set(result) == set(range(len(ancestor_ids)))


def test_non_contiguous_ids():
    """Test that non-contiguous IDs raise NotImplementedError."""
    df = pl.DataFrame(
        {
            "id": [10, 20, 30, 40, 50],
            "ancestor_id": [10, 10, 10, 20, 20],
        },
    )
    with pytest.raises(NotImplementedError):
        alifestd_unfurl_traversal_inorder_polars(df)


def test_non_topologically_sorted():
    """Test that non-topologically-sorted data raises NotImplementedError."""
    df = pl.DataFrame(
        {
            "id": [0, 1, 2, 3, 4],
            "ancestor_id": [1, 1, 1, 0, 0],
        },
    )
    with pytest.raises(NotImplementedError):
        alifestd_unfurl_traversal_inorder_polars(df)


def test_with_precomputed_columns():
    """Test that pre-existing num_leaves and right_child_id are reused."""
    ancestor_ids = np.array([0, 0, 0, 1, 1, 2, 2], dtype=np.int64)
    df = _make_contiguous_df(ancestor_ids)
    df = df.with_columns(
        num_leaves=pl.Series([4, 2, 2, 1, 1, 1, 1]),
        right_child_id=pl.Series([2, 4, 6, 3, 4, 5, 6]),
    )
    result = alifestd_unfurl_traversal_inorder_polars(df)
    assert tuple(result.tolist()) == (3, 1, 4, 0, 5, 2, 6)


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_lazyframe(apply):
    """Test that LazyFrame input works."""
    df = apply(
        _make_contiguous_df(np.array([0, 0, 0, 1, 1, 2, 2])),
    )
    result = alifestd_unfurl_traversal_inorder_polars(df)
    assert tuple(result.tolist()) == (3, 1, 4, 0, 5, 2, 6)


def test_with_ancestor_list_col():
    """Test with ancestor_list instead of ancestor_id."""
    df = pl.DataFrame(
        {
            "id": [0, 1, 2, 3, 4, 5, 6],
            "ancestor_list": [
                "[None]",
                "[0]",
                "[0]",
                "[1]",
                "[1]",
                "[2]",
                "[2]",
            ],
        },
    )
    result = alifestd_unfurl_traversal_inorder_polars(df)
    assert tuple(result.tolist()) == (3, 1, 4, 0, 5, 2, 6)


def test_matches_pandas():
    """Verify polars result matches pandas result."""
    import pandas as pd

    from phyloframe.legacy import alifestd_unfurl_traversal_inorder_asexual

    ancestor_ids = np.array([0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    n = len(ancestor_ids)

    pd_df = pd.DataFrame(
        {
            "id": np.arange(n),
            "ancestor_id": ancestor_ids,
        },
    )
    pl_df = _make_contiguous_df(ancestor_ids)

    pd_result = alifestd_unfurl_traversal_inorder_asexual(pd_df, mutate=True)
    pl_result = alifestd_unfurl_traversal_inorder_polars(pl_df)

    np.testing.assert_array_equal(pd_result, pl_result)
