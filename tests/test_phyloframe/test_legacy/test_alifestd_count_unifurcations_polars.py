import polars as pl

from phyloframe.legacy import (
    alifestd_count_unifurcations_polars as alifestd_count_unifurcations_polars_,
)

from ._impl import assert_dtype_consistency

alifestd_count_unifurcations_polars = assert_dtype_consistency(
    alifestd_count_unifurcations_polars_
)


def test_empty_df():
    df = pl.DataFrame(
        {"id": [], "ancestor_id": []},
        schema={"id": pl.Int64, "ancestor_id": pl.Int64},
    )
    assert alifestd_count_unifurcations_polars(df) == 0


def test_singleton_df():
    df = pl.DataFrame(
        {
            "id": [0],
            "ancestor_id": [0],
        }
    )
    assert alifestd_count_unifurcations_polars(df) == 0


def test_polytomy_df():
    df = pl.DataFrame(
        {
            "id": [0, 1, 2, 3, 4],
            "ancestor_id": [0, 0, 0, 0, 1],
        }
    )
    assert alifestd_count_unifurcations_polars(df) == 1


def test_no_unifurcations():
    df = pl.DataFrame(
        {
            "id": [0, 1, 2, 3, 4, 5],
            "ancestor_id": [0, 1, 0, 2, 2, 0],
        }
    )
    assert alifestd_count_unifurcations_polars(df) == 0


def test_all_unifurcations():
    df = pl.DataFrame(
        {
            "id": [0, 1, 2, 3, 4, 5],
            "ancestor_id": [0, 1, 0, 1, 2, 3],
        }
    )
    assert alifestd_count_unifurcations_polars(df) == 4


def test_strictly_bifurcating():
    df = pl.DataFrame(
        {
            "id": [0, 1, 2, 3, 4],
            "ancestor_id": [0, 0, 0, 1, 1],
        }
    )
    assert alifestd_count_unifurcations_polars(df) == 0
