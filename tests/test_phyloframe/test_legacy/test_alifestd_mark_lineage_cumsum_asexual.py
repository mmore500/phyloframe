import numpy as np
import pandas as pd
import pytest

from phyloframe.legacy import (
    alifestd_make_empty,
    alifestd_mark_lineage_cumsum_asexual,
)


def test_empty():
    mt = alifestd_make_empty()
    mt["origin_time"] = pd.Series(dtype=float)
    res = alifestd_mark_lineage_cumsum_asexual(mt, "origin_time")
    assert "lineage_cumsum" in res.columns
    assert len(res) == 0


def test_singleton():
    df = pd.DataFrame(
        {
            "id": [0],
            "ancestor_list": ["[None]"],
            "origin_time": [42.0],
        }
    )
    res = alifestd_mark_lineage_cumsum_asexual(df, "origin_time")
    assert res["lineage_cumsum"].tolist() == [42.0]
    res_r = alifestd_mark_lineage_cumsum_asexual(
        df, "origin_time", reverse=True
    )
    assert res_r["lineage_cumsum"].tolist() == [42.0]


@pytest.mark.parametrize("mutate", [True, False])
def test_simple_tree(mutate: bool):
    """Tree:
    0 (ot=10)
    +-- 1 (ot=1)
    |   +-- 3 (ot=3)
    |   +-- 4 (ot=4)
    +-- 2 (ot=2)
    """
    df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3, 4],
            "ancestor_list": ["[None]", "[0]", "[0]", "[1]", "[1]"],
            "origin_time": [10.0, 1.0, 2.0, 3.0, 4.0],
        }
    )
    original = df.copy()
    res = alifestd_mark_lineage_cumsum_asexual(
        df, "origin_time", mutate=mutate
    )
    assert res.set_index("id")["lineage_cumsum"].to_dict() == {
        0: 10.0,
        1: 11.0,
        2: 12.0,
        3: 14.0,
        4: 15.0,
    }
    if not mutate:
        assert original.equals(df)


def test_reverse():
    df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3, 4],
            "ancestor_list": ["[None]", "[0]", "[0]", "[1]", "[1]"],
            "origin_time": [10.0, 1.0, 2.0, 3.0, 4.0],
        }
    )
    res = alifestd_mark_lineage_cumsum_asexual(df, "origin_time", reverse=True)
    assert res.set_index("id")["lineage_cumsum"].to_dict() == {
        0: 20.0,
        1: 8.0,
        2: 2.0,
        3: 3.0,
        4: 4.0,
    }


def test_forest():
    df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "ancestor_list": ["[None]", "[None]", "[0]", "[1]"],
            "origin_time": [10.0, 20.0, 1.0, 2.0],
        }
    )
    res = alifestd_mark_lineage_cumsum_asexual(df, "origin_time")
    assert res.set_index("id")["lineage_cumsum"].to_dict() == {
        0: 10.0,
        1: 20.0,
        2: 11.0,
        3: 22.0,
    }
    res_r = alifestd_mark_lineage_cumsum_asexual(
        df, "origin_time", reverse=True
    )
    assert res_r.set_index("id")["lineage_cumsum"].to_dict() == {
        0: 11.0,
        1: 22.0,
        2: 1.0,
        3: 2.0,
    }


def test_mark_as():
    df = pd.DataFrame(
        {
            "id": [0, 1],
            "ancestor_list": ["[None]", "[0]"],
            "origin_time": [10.0, 5.0],
        }
    )
    res = alifestd_mark_lineage_cumsum_asexual(
        df, "origin_time", mark_as="custom_col"
    )
    assert "custom_col" in res.columns
    assert "lineage_cumsum" not in res.columns


def test_skipna():
    df = pd.DataFrame(
        {
            "id": [0, 1, 2],
            "ancestor_list": ["[None]", "[0]", "[1]"],
            "v": [1.0, np.nan, 3.0],
        }
    )
    res = alifestd_mark_lineage_cumsum_asexual(df, "v", skipna=True)
    assert res["lineage_cumsum"].tolist() == [1.0, 1.0, 4.0]
    res_no = alifestd_mark_lineage_cumsum_asexual(df, "v", skipna=False)
    out = res_no["lineage_cumsum"].tolist()
    assert out[0] == 1.0
    assert np.isnan(out[1])
    assert np.isnan(out[2])


def test_not_asexual():
    df = pd.DataFrame(
        {
            "id": [0, 1, 2],
            "ancestor_list": ["[None]", "[0]", "[0,1]"],
            "origin_time": [0.0, 1.0, 2.0],
        }
    )
    with pytest.raises(NotImplementedError):
        alifestd_mark_lineage_cumsum_asexual(df, "origin_time")


def test_non_contiguous_ids():
    df = pd.DataFrame(
        {
            "id": [0, 2, 5],
            "ancestor_id": [0, 0, 2],
            "ancestor_list": ["[None]", "[0]", "[2]"],
            "origin_time": [0.0, 1.0, 2.0],
        }
    )
    with pytest.raises(NotImplementedError):
        alifestd_mark_lineage_cumsum_asexual(df, "origin_time")


def test_unsorted():
    df = pd.DataFrame(
        {
            "id": [0, 1, 2],
            "ancestor_id": [0, 2, 0],
            "ancestor_list": ["[None]", "[2]", "[0]"],
            "origin_time": [0.0, 1.0, 2.0],
        }
    )
    with pytest.raises(NotImplementedError):
        alifestd_mark_lineage_cumsum_asexual(df, "origin_time")


def test_missing_values_column():
    df = pd.DataFrame(
        {
            "id": [0, 1],
            "ancestor_list": ["[None]", "[0]"],
            "origin_time": [0.0, 1.0],
        }
    )
    with pytest.raises(ValueError):
        alifestd_mark_lineage_cumsum_asexual(df, "no_such_col")
