import numpy as np
import pandas as pd
import pytest

from phyloframe.legacy import (
    alifestd_make_empty,
    alifestd_mark_lineage_cumprod_asexual,
)


def test_empty():
    mt = alifestd_make_empty()
    mt["v"] = pd.Series(dtype=float)
    res = alifestd_mark_lineage_cumprod_asexual(mt, "v")
    assert "lineage_cumprod" in res.columns
    assert len(res) == 0


def test_singleton():
    df = pd.DataFrame(
        {
            "id": [0],
            "ancestor_list": ["[None]"],
            "v": [3.0],
        }
    )
    res = alifestd_mark_lineage_cumprod_asexual(df, "v")
    assert res["lineage_cumprod"].tolist() == [3.0]


@pytest.mark.parametrize("mutate", [True, False])
def test_simple_tree(mutate: bool):
    df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "ancestor_list": ["[None]", "[0]", "[1]", "[2]"],
            "v": [2.0, 3.0, 5.0, 7.0],
        }
    )
    original = df.copy()
    res = alifestd_mark_lineage_cumprod_asexual(df, "v", mutate=mutate)
    assert res.set_index("id")["lineage_cumprod"].to_dict() == {
        0: 2.0,
        1: 6.0,
        2: 30.0,
        3: 210.0,
    }
    if not mutate:
        assert original.equals(df)


def test_reverse():
    df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "ancestor_list": ["[None]", "[0]", "[0]", "[0]"],
            "v": [2.0, 3.0, 5.0, 7.0],
        }
    )
    res = alifestd_mark_lineage_cumprod_asexual(df, "v", reverse=True)
    # clade product at root = 2 * 3 * 5 * 7 = 210
    assert res.set_index("id")["lineage_cumprod"].to_dict() == {
        0: 210.0,
        1: 3.0,
        2: 5.0,
        3: 7.0,
    }


def test_forest():
    df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "ancestor_list": ["[None]", "[None]", "[0]", "[1]"],
            "v": [2.0, 3.0, 4.0, 5.0],
        }
    )
    res = alifestd_mark_lineage_cumprod_asexual(df, "v")
    assert res.set_index("id")["lineage_cumprod"].to_dict() == {
        0: 2.0,
        1: 3.0,
        2: 8.0,
        3: 15.0,
    }


def test_skipna():
    df = pd.DataFrame(
        {
            "id": [0, 1, 2],
            "ancestor_list": ["[None]", "[0]", "[1]"],
            "v": [2.0, np.nan, 5.0],
        }
    )
    res = alifestd_mark_lineage_cumprod_asexual(df, "v", skipna=True)
    assert res["lineage_cumprod"].tolist() == [2.0, 2.0, 10.0]


def test_not_asexual():
    df = pd.DataFrame(
        {
            "id": [0, 1, 2],
            "ancestor_list": ["[None]", "[0]", "[0,1]"],
            "v": [1.0, 2.0, 3.0],
        }
    )
    with pytest.raises(NotImplementedError):
        alifestd_mark_lineage_cumprod_asexual(df, "v")


def test_non_contiguous_ids():
    df = pd.DataFrame(
        {
            "id": [0, 2, 5],
            "ancestor_id": [0, 0, 2],
            "ancestor_list": ["[None]", "[0]", "[2]"],
            "v": [1.0, 2.0, 3.0],
        }
    )
    with pytest.raises(NotImplementedError):
        alifestd_mark_lineage_cumprod_asexual(df, "v")


def test_unsorted():
    df = pd.DataFrame(
        {
            "id": [0, 1, 2],
            "ancestor_id": [0, 2, 0],
            "ancestor_list": ["[None]", "[2]", "[0]"],
            "v": [1.0, 2.0, 3.0],
        }
    )
    with pytest.raises(NotImplementedError):
        alifestd_mark_lineage_cumprod_asexual(df, "v")


def test_missing_values_column():
    df = pd.DataFrame(
        {
            "id": [0, 1],
            "ancestor_list": ["[None]", "[0]"],
            "v": [0.0, 1.0],
        }
    )
    with pytest.raises(ValueError):
        alifestd_mark_lineage_cumprod_asexual(df, "no_such_col")
