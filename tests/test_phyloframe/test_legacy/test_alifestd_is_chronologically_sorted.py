import pandas as pd

from phyloframe.legacy import (
    alifestd_is_chronologically_sorted as alifestd_is_chronologically_sorted_,
)
from phyloframe.legacy import (
    alifestd_make_empty,
)

from ._impl import assert_dtype_consistency

alifestd_is_chronologically_sorted = assert_dtype_consistency(alifestd_is_chronologically_sorted_)


def test_is_chronologically_sorted_empty():
    df = alifestd_make_empty()
    df["origin_time"] = []
    assert alifestd_is_chronologically_sorted(df)


def test_is_chronologically_sorted_missing_col():
    phylogeny_df = pd.DataFrame(
        {
            "id": [0, 1, 22, 333, 44, 5],
        }
    )
    assert not alifestd_is_chronologically_sorted(phylogeny_df)


def test_is_chronologically_sorted():
    phylogeny_df = pd.DataFrame(
        {
            "id": [0, 1, 22, 333, 44, 5],
            "origin_time": [0, 1, 2, 2, 5, 6],
        }
    )
    assert alifestd_is_chronologically_sorted(phylogeny_df)


def test_is_not_chronologically_sorted():
    phylogeny_df = pd.DataFrame(
        {
            "id": [0, 1, 22, 333, 44, 5],
            "origin_time": [2, 1, 2, 2, 5, 6],
        }
    )
    assert not alifestd_is_chronologically_sorted(phylogeny_df)


def test_is_chronologically_sorted_destruction_time():
    phylogeny_df = pd.DataFrame(
        {
            "id": [0, 1, 22, 333, 44, 5],
            "destruction_time": [0, 1, 2, 2, 5, 6],
        }
    )
    assert alifestd_is_chronologically_sorted(
        phylogeny_df, how="destruction_time"
    )


def test_is_not_chronologically_sorted_destruction_time():
    phylogeny_df = pd.DataFrame(
        {
            "id": [0, 1, 22, 333, 44, 5],
            "destruction_time": [2, 1, 2, 2, 5, 6],
        }
    )
    assert not alifestd_is_chronologically_sorted(
        phylogeny_df, how="destruction_time"
    )
