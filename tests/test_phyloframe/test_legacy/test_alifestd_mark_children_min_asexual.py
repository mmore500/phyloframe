import numpy as np
import pandas as pd
import pytest

from phyloframe.legacy import alifestd_mark_children_min_asexual


def _make_df() -> pd.DataFrame:
    """Tree:
    0 (root)
    +-- 1
    |   +-- 2 (v=NaN)
    +-- 3
    4 (root)
    """
    return pd.DataFrame(
        {
            "id": [0, 1, 2, 3, 4],
            "ancestor_id": [0, 0, 1, 0, 4],
            "v": [0.0, 10.0, np.nan, 30.0, 7.0],
        },
    )


@pytest.mark.parametrize("mutate", [True, False])
def test_simple_tree(mutate: bool):
    df = _make_df()
    original_df = df.copy()
    res = alifestd_mark_children_min_asexual(df, "v", mutate=mutate)
    np.testing.assert_array_equal(
        res["children_min"], [10.0, np.nan, np.nan, np.nan, np.nan]
    )
    if not mutate:
        assert df.equals(original_df)


def test_skipna_false():
    df = _make_df()
    df.loc[3, "v"] = np.nan
    res = alifestd_mark_children_min_asexual(df, "v", skipna=False)
    assert np.isnan(res["children_min"]).all()

    res = alifestd_mark_children_min_asexual(df, "v")
    np.testing.assert_array_equal(
        res["children_min"], [10.0, np.nan, np.nan, np.nan, np.nan]
    )


def test_unsorted():
    # 2 <- 0, 2 <- 1, 2 root
    df = pd.DataFrame(
        {"id": [0, 1, 2], "ancestor_id": [2, 2, 2], "v": [5.0, 3.0, 1.0]}
    )
    res = alifestd_mark_children_min_asexual(df, "v")
    np.testing.assert_array_equal(res["children_min"], [np.nan, np.nan, 3.0])


def test_mark_as():
    res = alifestd_mark_children_min_asexual(
        _make_df(), "v", mark_as="blueberry"
    )
    assert "blueberry" in res.columns
    assert "children_min" not in res.columns


def test_empty():
    df = pd.DataFrame({"id": [], "ancestor_id": [], "v": []}, dtype=float)
    res = alifestd_mark_children_min_asexual(df, "v")
    assert "children_min" in res.columns
    assert len(res) == 0


def test_missing_column():
    with pytest.raises(ValueError):
        alifestd_mark_children_min_asexual(_make_df(), "nope")


def test_non_contiguous_ids():
    df = pd.DataFrame({"id": [0, 2], "ancestor_id": [0, 0], "v": [1.0, 1.0]})
    with pytest.raises(NotImplementedError):
        alifestd_mark_children_min_asexual(df, "v")
