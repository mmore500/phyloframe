import numpy as np
import pandas as pd
import pytest

from phyloframe.legacy import alifestd_find_oldest_root_id_asexual


def test_empty():
    df = pd.DataFrame({"id": [], "ancestor_id": []}, dtype=int)
    assert alifestd_find_oldest_root_id_asexual(df) is None


def test_lowest_id_without_origin_time():
    df = pd.DataFrame({"id": [0, 1, 2, 3], "ancestor_id": [1, 1, 3, 3]})
    assert alifestd_find_oldest_root_id_asexual(df) == 1


def test_lowest_origin_time_ties_by_id():
    df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "ancestor_id": [0, 1, 2, 0],
            "origin_time": [3.0, 1.0, 1.0, 0.0],
        },
    )
    # node 3 is earliest but not a root; roots 1 and 2 tie
    assert alifestd_find_oldest_root_id_asexual(df) == 1


def test_ancestor_list():
    df = pd.DataFrame(
        {"id": [0, 1, 2], "ancestor_list": ["[2]", "[none]", "[none]"]}
    )
    original = df.copy()
    assert alifestd_find_oldest_root_id_asexual(df) == 1
    pd.testing.assert_frame_equal(df, original)


@pytest.mark.parametrize("bad", [None, np.nan])
def test_null_or_nan_root_origin_time_raises(bad):
    df = pd.DataFrame(
        {"id": [0, 1], "ancestor_id": [0, 1], "origin_time": [bad, 1.0]},
    )
    with pytest.raises(NotImplementedError, match="origin_time"):
        alifestd_find_oldest_root_id_asexual(df)


def test_nan_nonroot_origin_time_ok():
    df = pd.DataFrame(
        {
            "id": [0, 1, 2],
            "ancestor_id": [0, 1, 0],
            "origin_time": [2.0, 1.0, np.nan],
        },
    )
    assert alifestd_find_oldest_root_id_asexual(df) == 1
