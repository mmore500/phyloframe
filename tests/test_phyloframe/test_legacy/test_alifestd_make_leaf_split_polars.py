import pytest

from phyloframe.legacy._alifestd_make_leaf_split_polars import (
    alifestd_make_leaf_split_polars,
)


def test_make_leaf_split_polars_0_leaves():
    result = alifestd_make_leaf_split_polars(0, seed=0)
    assert result.is_empty()
    assert "id" in result.columns
    assert "ancestor_id" in result.columns


def test_make_leaf_split_polars_1_leaf():
    result = alifestd_make_leaf_split_polars(1, seed=0)
    assert len(result) == 1
    assert result["id"].to_list() == [0]
    assert result["ancestor_id"].to_list() == [0]


def test_make_leaf_split_polars_2_leaves():
    result = alifestd_make_leaf_split_polars(2, seed=0)
    assert len(result) == 3


@pytest.mark.parametrize("n_leaves", [3, 4, 5, 8, 16])
def test_make_leaf_split_polars_node_count(n_leaves: int):
    result = alifestd_make_leaf_split_polars(n_leaves, seed=42)
    assert len(result) == 2 * n_leaves - 1


@pytest.mark.parametrize("n_leaves", [2, 3, 4, 5, 8, 16])
@pytest.mark.parametrize("seed", [0, 1, 42])
def test_make_leaf_split_polars_deterministic(n_leaves: int, seed: int):
    a = alifestd_make_leaf_split_polars(n_leaves, seed=seed)
    b = alifestd_make_leaf_split_polars(n_leaves, seed=seed)
    assert a.equals(b)


@pytest.mark.parametrize("n_leaves", [5, 8, 16])
def test_make_leaf_split_polars_no_seed(n_leaves: int):
    result = alifestd_make_leaf_split_polars(n_leaves)
    assert len(result) == 2 * n_leaves - 1


def test_make_leaf_split_polars_negative():
    with pytest.raises(ValueError):
        alifestd_make_leaf_split_polars(-1, seed=0)
