import os

import numpy as np
import polars as pl
import pytest

from phyloframe.legacy import (
    alifestd_unfurl_traversal_semiorder_polars,
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
    result = alifestd_unfurl_traversal_semiorder_polars(df)
    assert len(result) == 0


def test_not_strictly_bifurcating():
    """Chain tree (unifurcating) should raise ValueError."""
    df = _make_contiguous_df(np.array([0, 0, 1]))
    with pytest.raises(ValueError):
        alifestd_unfurl_traversal_semiorder_polars(df)


def test_multiple_roots():
    """Multiple roots should raise ValueError."""
    df = _make_contiguous_df(
        np.array([0, 1, 0, 0, 1, 1]),
    )
    with pytest.raises(ValueError):
        alifestd_unfurl_traversal_semiorder_polars(df)


def test_simple_balanced():
    """Tree structure:
            0
          /   \\
         1     2
        / \\   / \\
       3   4 5   6

    Semiorder: either child can be visited first.
    """
    df = _make_contiguous_df(np.array([0, 0, 0, 1, 1, 2, 2]))
    result = alifestd_unfurl_traversal_semiorder_polars(df)
    valid_orders = {
        (3, 1, 4, 0, 5, 2, 6),
        (4, 1, 3, 0, 5, 2, 6),
        (3, 1, 4, 0, 6, 2, 5),
        (4, 1, 3, 0, 5, 2, 5),
        (5, 2, 6, 0, 3, 1, 4),
        (5, 2, 6, 0, 4, 1, 3),
        (6, 2, 5, 0, 3, 1, 4),
        (5, 2, 5, 0, 4, 1, 3),
    }
    assert tuple(result.tolist()) in valid_orders


def test_left_heavy():
    """Tree structure:
            0
          /   \\
         1     2
        / \\
       3   4
          / \\
         5   6

    Semiorder: either child can be visited first.
    """
    df = _make_contiguous_df(np.array([0, 0, 0, 1, 1, 4, 4]))
    result = alifestd_unfurl_traversal_semiorder_polars(df)
    valid_orders = {
        (5, 4, 6, 1, 3, 0, 2),
        (6, 4, 5, 1, 3, 0, 2),
        (3, 1, 5, 4, 6, 0, 2),
        (3, 1, 6, 4, 5, 0, 2),
        (2, 0, 5, 4, 6, 1, 3),
        (2, 0, 6, 4, 5, 1, 3),
        (2, 0, 3, 1, 5, 4, 6),
        (2, 0, 3, 1, 6, 4, 5),
    }
    assert tuple(result.tolist()) in valid_orders


def test_permutation():
    """Verify result is a valid permutation of node ids."""
    ancestor_ids = np.array([0, 0, 0, 1, 1, 2, 2], dtype=np.int64)
    df = _make_contiguous_df(ancestor_ids)
    result = alifestd_unfurl_traversal_semiorder_polars(df)
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
        alifestd_unfurl_traversal_semiorder_polars(df)


def test_non_topologically_sorted():
    """Test that non-topologically-sorted data raises NotImplementedError."""
    df = pl.DataFrame(
        {
            "id": [0, 1, 2, 3, 4],
            "ancestor_id": [1, 1, 1, 0, 0],
        },
    )
    with pytest.raises((NotImplementedError, ValueError)):
        alifestd_unfurl_traversal_semiorder_polars(df)


def test_with_precomputed_num_descendants():
    """Test that pre-existing num_descendants column is reused."""
    ancestor_ids = np.array([0, 0, 0, 1, 1, 2, 2], dtype=np.int64)
    df = _make_contiguous_df(ancestor_ids)
    df = df.with_columns(
        num_descendants=pl.Series([6, 2, 2, 0, 0, 0, 0]),
    )
    result = alifestd_unfurl_traversal_semiorder_polars(df)
    assert len(result) == len(ancestor_ids)
    assert set(result) == set(range(len(ancestor_ids)))


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
    result = alifestd_unfurl_traversal_semiorder_polars(df)
    assert len(result) == 7
    assert set(result) == set(range(7))


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
    result = alifestd_unfurl_traversal_semiorder_polars(df)
    assert len(result) == 7
    assert set(result) == set(range(7))


def test_matches_pandas():
    """Verify polars result matches pandas result."""
    import pandas as pd

    from phyloframe.legacy import (
        alifestd_unfurl_traversal_semiorder_asexual,
    )

    ancestor_ids = np.array([0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    n = len(ancestor_ids)

    pd_df = pd.DataFrame(
        {
            "id": np.arange(n),
            "ancestor_id": ancestor_ids,
        },
    )
    pl_df = _make_contiguous_df(ancestor_ids)

    pd_result = alifestd_unfurl_traversal_semiorder_asexual(pd_df, mutate=True)
    pl_result = alifestd_unfurl_traversal_semiorder_polars(pl_df)

    np.testing.assert_array_equal(pd_result, pl_result)
