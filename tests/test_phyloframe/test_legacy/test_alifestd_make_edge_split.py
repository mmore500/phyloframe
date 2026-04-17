import pandas as pd
import pytest

from phyloframe.legacy import (
    alifestd_find_leaf_ids,
    alifestd_is_strictly_bifurcating_asexual,
    alifestd_is_topologically_sorted,
    alifestd_make_edge_split,
    alifestd_validate,
)


@pytest.mark.parametrize("n_leaves", [2, 3, 4, 5, 8, 16])
def test_node_count(n_leaves: int):
    """Total node count should be 2*n_leaves - 1."""
    df = alifestd_make_edge_split(n_leaves, seed=42)
    assert len(df) == 2 * n_leaves - 1


@pytest.mark.parametrize("n_leaves", [2, 3, 4, 5, 8, 16])
def test_leaf_count(n_leaves: int):
    df = alifestd_make_edge_split(n_leaves, seed=42)
    leaf_ids = [*alifestd_find_leaf_ids(df)]
    assert len(leaf_ids) == n_leaves


@pytest.mark.parametrize("n_leaves", [2, 3, 4, 5, 8, 16])
def test_contiguous_ids(n_leaves: int):
    """IDs should be contiguous starting from 0."""
    df = alifestd_make_edge_split(n_leaves, seed=42)
    assert list(df["id"]) == list(range(len(df)))


@pytest.mark.parametrize("n_leaves", [1, 2, 3, 4, 5, 8, 16])
@pytest.mark.parametrize("seed", [0, 1, 42, 12345])
def test_validates(n_leaves: int, seed: int):
    df = alifestd_make_edge_split(n_leaves, seed=seed)
    assert alifestd_validate(df)


@pytest.mark.parametrize("n_leaves", [2, 3, 4, 5, 8, 16])
def test_bifurcating_structure(n_leaves: int):
    df = alifestd_make_edge_split(n_leaves, seed=42)
    assert alifestd_is_strictly_bifurcating_asexual(df)


@pytest.mark.parametrize("n_leaves", [2, 3, 4, 5, 8, 16])
def test_topological_sorting(n_leaves: int):
    df = alifestd_make_edge_split(n_leaves, seed=42)
    assert alifestd_is_topologically_sorted(df)


@pytest.mark.parametrize("n_leaves", [1, 2, 3, 4, 5, 8])
def test_returns_dataframe(n_leaves: int):
    result = alifestd_make_edge_split(n_leaves, seed=7)
    assert isinstance(result, pd.DataFrame)
    assert "id" in result.columns
    assert "ancestor_list" in result.columns


@pytest.mark.parametrize("n_leaves", [2, 3, 4, 5, 8, 16])
@pytest.mark.parametrize("seed", [0, 1, 42, 12345])
def test_deterministic(n_leaves: int, seed: int):
    a = alifestd_make_edge_split(n_leaves, seed=seed)
    b = alifestd_make_edge_split(n_leaves, seed=seed)
    assert a.equals(b)


@pytest.mark.parametrize("n_leaves", [5, 8, 16])
def test_different_seeds_differ(n_leaves: int):
    """Different seeds should usually produce different trees."""
    a = alifestd_make_edge_split(n_leaves, seed=1)
    b = alifestd_make_edge_split(n_leaves, seed=2)
    assert not a.equals(b)


@pytest.mark.parametrize("n_leaves", [5, 8, 16])
def test_no_seed(n_leaves: int):
    """Omitting seed should still produce a valid tree."""
    df = alifestd_make_edge_split(n_leaves)
    assert len(df) == 2 * n_leaves - 1


def test_seed_does_not_leak_rng_state():
    """Using a seed should not perturb outer RNG state."""
    import random

    import numpy as np

    random.seed(0)
    np.random.seed(0)
    expected_py = random.random()
    expected_np = np.random.random()

    random.seed(0)
    np.random.seed(0)
    alifestd_make_edge_split(16, seed=42)
    assert random.random() == expected_py
    assert np.random.random() == expected_np


def test_zero_leaves():
    df = alifestd_make_edge_split(0, seed=0)
    assert len(df) == 0
    assert isinstance(df, pd.DataFrame)


def test_single_leaf():
    df = alifestd_make_edge_split(1, seed=0)
    assert len(df) == 1
    assert df.iloc[0]["ancestor_list"] == "[None]"


def test_negative_leaves():
    with pytest.raises(ValueError):
        alifestd_make_edge_split(-1, seed=0)
