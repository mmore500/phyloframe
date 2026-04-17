import pandas as pd
import pytest

from phyloframe.legacy import (
    alifestd_find_leaf_ids,
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
    """Every internal node should have exactly 2 children."""
    df = alifestd_make_edge_split(n_leaves, seed=42)
    leaf_ids = set(alifestd_find_leaf_ids(df))
    for _, row in df.iterrows():
        if row["id"] not in leaf_ids:
            children = df[df["ancestor_list"] == f"[{row['id']}]"]
            assert len(children) == 2


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
