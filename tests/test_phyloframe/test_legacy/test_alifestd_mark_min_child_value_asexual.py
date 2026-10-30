import numpy as np
import pandas as pd
import pytest

from phyloframe.legacy import alifestd_mark_min_child_value_asexual


def _make_df() -> pd.DataFrame:
    """Tree:
    0 (root)
    +-- 1
    |   +-- 2
    +-- 3
    4 (root)
    """
    return pd.DataFrame(
        {
            "id": [0, 1, 2, 3, 4],
            "ancestor_id": [0, 0, 1, 0, 4],
            "v": [0.0, 10.0, 20.0, 30.0, 7.0],
        },
    )


@pytest.mark.parametrize("mutate", [True, False])
def test_simple_tree(mutate: bool):
    df = _make_df()
    original_df = df.copy()
    res = alifestd_mark_min_child_value_asexual(df, "v", mutate=mutate)
    # leaves keep their own value
    np.testing.assert_array_equal(
        res["min_child_value"], [10.0, 20.0, 20.0, 30.0, 7.0]
    )
    if not mutate:
        assert df.equals(original_df)


def test_children_larger_than_parent():
    # 0 <- 1, 2
    df = pd.DataFrame(
        {"id": [0, 1, 2], "ancestor_id": [0, 0, 0], "v": [-5.0, 3.0, 4.0]}
    )
    res = alifestd_mark_min_child_value_asexual(df, "v")
    np.testing.assert_array_equal(res["min_child_value"], [3.0, 3.0, 4.0])


def test_unsorted():
    # 2 <- 0, 2 <- 1, 2 root
    df = pd.DataFrame(
        {"id": [0, 1, 2], "ancestor_id": [2, 2, 2], "v": [5.0, 3.0, 1.0]}
    )
    res = alifestd_mark_min_child_value_asexual(df, "v")
    np.testing.assert_array_equal(res["min_child_value"], [5.0, 3.0, 3.0])


def test_mark_as():
    res = alifestd_mark_min_child_value_asexual(
        _make_df(), "v", mark_as="blueberry"
    )
    assert "blueberry" in res.columns
    assert "min_child_value" not in res.columns


def test_nan_not_implemented():
    df = _make_df()
    df.loc[2, "v"] = np.nan
    with pytest.raises(NotImplementedError):
        alifestd_mark_min_child_value_asexual(df, "v")


def test_empty():
    df = pd.DataFrame({"id": [], "ancestor_id": [], "v": []}, dtype=float)
    res = alifestd_mark_min_child_value_asexual(df, "v")
    assert "min_child_value" in res.columns
    assert len(res) == 0


def test_missing_column():
    with pytest.raises(KeyError):
        alifestd_mark_min_child_value_asexual(_make_df(), "nope")


def test_non_contiguous_ids():
    df = pd.DataFrame({"id": [0, 2], "ancestor_id": [0, 0], "v": [1.0, 1.0]})
    with pytest.raises(NotImplementedError):
        alifestd_mark_min_child_value_asexual(df, "v")


def test_integer_dtype_preserved():
    df = _make_df().astype({"v": int})
    res = alifestd_mark_min_child_value_asexual(df, "v")
    assert np.issubdtype(res["min_child_value"].dtype, np.integer)
    np.testing.assert_array_equal(res["min_child_value"], [10, 20, 20, 30, 7])
