import pandas as pd
import pytest

from phyloframe.legacy import (
    alifestd_find_leaf_ids,
    alifestd_make_star,
    alifestd_validate,
)


@pytest.mark.parametrize("n_leaves", [2, 3, 4, 5, 8])
def test_node_count(n_leaves: int):
    """Total node count should be n_leaves + 1."""
    df = alifestd_make_star(n_leaves)
    assert len(df) == n_leaves + 1


@pytest.mark.parametrize("n_leaves", [2, 3, 4, 5, 8])
def test_leaf_count(n_leaves: int):
    df = alifestd_make_star(n_leaves)
    leaf_ids = [*alifestd_find_leaf_ids(df)]
    assert len(leaf_ids) == n_leaves


@pytest.mark.parametrize("n_leaves", [2, 3, 4, 5, 8])
def test_contiguous_ids(n_leaves: int):
    """IDs should be contiguous starting from 0."""
    df = alifestd_make_star(n_leaves)
    assert list(df["id"]) == list(range(len(df)))


@pytest.mark.parametrize("n_leaves", [2, 3, 4, 5, 8])
def test_validates(n_leaves: int):
    df = alifestd_make_star(n_leaves)
    assert alifestd_validate(df)


@pytest.mark.parametrize("n_leaves", [2, 3, 4, 5, 8])
def test_star_structure(n_leaves: int):
    """Every non-root node should be a direct child of the root."""
    df = alifestd_make_star(n_leaves)
    assert df.iloc[0]["ancestor_list"] == "[None]"
    for _, row in df.iloc[1:].iterrows():
        assert row["ancestor_list"] == "[0]"


@pytest.mark.parametrize("n_leaves", [2, 3, 4, 5, 8])
def test_root_has_all_leaves_as_children(n_leaves: int):
    """The root should have exactly n_leaves children."""
    df = alifestd_make_star(n_leaves)
    children = df[df["ancestor_list"] == "[0]"]
    assert len(children) == n_leaves


@pytest.mark.parametrize("n_leaves", [2, 3, 4, 5, 8])
def test_topological_sorting(n_leaves: int):
    """Parents should appear before children (topologically sorted)."""
    df = alifestd_make_star(n_leaves)
    seen = set()
    for _, row in df.iterrows():
        if row["ancestor_list"] != "[None]":
            parent_id = int(row["ancestor_list"].strip("[]"))
            assert parent_id in seen
        seen.add(row["id"])


@pytest.mark.parametrize("n_leaves", [2, 3, 4, 5])
def test_returns_dataframe(n_leaves: int):
    result = alifestd_make_star(n_leaves)
    assert isinstance(result, pd.DataFrame)
    assert "id" in result.columns
    assert "ancestor_list" in result.columns


def test_zero_leaves():
    """n_leaves=0 should produce an empty tree."""
    df = alifestd_make_star(0)
    assert len(df) == 0
    assert isinstance(df, pd.DataFrame)


def test_negative_leaves():
    """Negative n_leaves should raise ValueError."""
    with pytest.raises(ValueError):
        alifestd_make_star(-1)


def test_single_leaf():
    """n_leaves=1 should produce a single root node."""
    df = alifestd_make_star(1)
    assert len(df) == 1
    assert df.iloc[0]["ancestor_list"] == "[None]"
