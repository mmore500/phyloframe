import numpy as np
import pandas as pd
import pytest

from phyloframe.legacy import (
    alifestd_make_empty,
    alifestd_mark_lineage_max_asexual,
)


def test_empty():
    mt = alifestd_make_empty()
    mt["v"] = pd.Series(dtype=float)
    res = alifestd_mark_lineage_max_asexual(mt, "v")
    assert "lineage_max" in res.columns
    assert len(res) == 0


def test_singleton():
    df = pd.DataFrame(
        {
            "id": [0],
            "ancestor_list": ["[None]"],
            "v": [7.0],
        }
    )
    res = alifestd_mark_lineage_max_asexual(df, "v")
    assert res["lineage_max"].tolist() == [7.0]


@pytest.mark.parametrize("mutate", [True, False])
def test_simple_tree(mutate: bool):
    df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3, 4],
            "ancestor_list": ["[None]", "[0]", "[0]", "[1]", "[1]"],
            "v": [10.0, 5.0, 20.0, 8.0, 1.0],
        }
    )
    original = df.copy()
    res = alifestd_mark_lineage_max_asexual(df, "v", mutate=mutate)
    assert res.set_index("id")["lineage_max"].to_dict() == {
        0: 10.0,
        1: 10.0,
        2: 20.0,
        3: 10.0,
        4: 10.0,
    }
    if not mutate:
        assert original.equals(df)


def test_reverse_clade_max():
    df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3, 4],
            "ancestor_list": ["[None]", "[0]", "[0]", "[1]", "[1]"],
            "v": [10.0, 5.0, 20.0, 8.0, 1.0],
        }
    )
    res = alifestd_mark_lineage_max_asexual(df, "v", reverse=True)
    assert res.set_index("id")["lineage_max"].to_dict() == {
        0: 20.0,
        1: 8.0,
        2: 20.0,
        3: 8.0,
        4: 1.0,
    }


def test_forest():
    df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "ancestor_list": ["[None]", "[None]", "[0]", "[1]"],
            "v": [10.0, 20.0, 5.0, 30.0],
        }
    )
    res = alifestd_mark_lineage_max_asexual(df, "v")
    assert res.set_index("id")["lineage_max"].to_dict() == {
        0: 10.0,
        1: 20.0,
        2: 10.0,
        3: 30.0,
    }


def test_skipna():
    df = pd.DataFrame(
        {
            "id": [0, 1, 2],
            "ancestor_list": ["[None]", "[0]", "[1]"],
            "v": [3.0, np.nan, 5.0],
        }
    )
    res = alifestd_mark_lineage_max_asexual(df, "v", skipna=True)
    assert res["lineage_max"].tolist() == [3.0, 3.0, 5.0]


def test_not_asexual():
    df = pd.DataFrame(
        {
            "id": [0, 1, 2],
            "ancestor_list": ["[None]", "[0]", "[0,1]"],
            "v": [1.0, 2.0, 3.0],
        }
    )
    with pytest.raises(NotImplementedError):
        alifestd_mark_lineage_max_asexual(df, "v")
