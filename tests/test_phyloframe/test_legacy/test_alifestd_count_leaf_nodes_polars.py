import polars as pl

from phyloframe.legacy import (
    alifestd_count_leaf_nodes_polars as alifestd_count_leaf_nodes_polars_,
)

from ._impl import assert_dtype_consistency

alifestd_count_leaf_nodes_polars = assert_dtype_consistency(alifestd_count_leaf_nodes_polars_)


def test_empty_df():
    df = pl.DataFrame(
        {"id": [], "ancestor_id": []},
        schema={"id": pl.Int64, "ancestor_id": pl.Int64},
    )
    assert alifestd_count_leaf_nodes_polars(df) == 0


def test_singleton_df():
    df = pl.DataFrame(
        {
            "id": [0],
            "ancestor_id": [0],
        }
    )
    assert alifestd_count_leaf_nodes_polars(df) == 1


def test_one_root_two_children():
    df = pl.DataFrame(
        {
            "id": [0, 1, 2],
            "ancestor_id": [0, 0, 0],
        }
    )
    assert alifestd_count_leaf_nodes_polars(df) == 2


def test_strictly_bifurcating_df():
    df = pl.DataFrame(
        {
            "id": [0, 1, 2, 3, 4],
            "ancestor_id": [0, 0, 0, 1, 1],
        }
    )
    assert alifestd_count_leaf_nodes_polars(df) == 3


def test_polytomy_df():
    df = pl.DataFrame(
        {
            "id": [0, 1, 2, 3, 4],
            "ancestor_id": [0, 0, 0, 0, 1],
        }
    )
    assert alifestd_count_leaf_nodes_polars(df) == 3


def test_multiple_trees_df1():
    df = pl.DataFrame(
        {
            "id": [0, 1, 2, 3, 4, 5],
            "ancestor_id": [0, 1, 0, 2, 2, 3],
        }
    )
    assert alifestd_count_leaf_nodes_polars(df) == 3


def test_multiple_trees_df2():
    df = pl.DataFrame(
        {
            "id": [0, 1, 2, 3, 4, 5],
            "ancestor_id": [0, 1, 0, 1, 2, 3],
        }
    )
    assert alifestd_count_leaf_nodes_polars(df) == 2
