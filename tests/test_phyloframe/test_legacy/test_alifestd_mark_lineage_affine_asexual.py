import numpy as np
import pandas as pd
import pytest

from phyloframe.legacy import (
    alifestd_mark_lineage_affine_asexual,
    alifestd_mark_lineage_cumprod_asexual,
    alifestd_mark_lineage_cumsum_asexual,
)


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
            "m": [9.0, 2.0, 3.0, 0.5, 9.0],
            "a": [1.0, 1.0, 1.0, 2.0, 5.0],
        },
    )


@pytest.mark.parametrize("mutate", [True, False])
def test_simple_tree(mutate: bool):
    df = _make_df()
    original_df = df.copy()
    res = alifestd_mark_lineage_affine_asexual(df, "m", "a", mutate=mutate)
    np.testing.assert_allclose(
        res["lineage_affine"], [1.0, 3.0, 10.0, 2.5, 5.0]
    )
    if not mutate:
        assert df.equals(original_df)


def test_mark_as():
    res = alifestd_mark_lineage_affine_asexual(
        _make_df(), "m", "a", mark_as="blueberry"
    )
    assert "blueberry" in res.columns
    assert "lineage_affine" not in res.columns


def test_nan_propagates():
    df = _make_df()
    df.loc[1, "a"] = np.nan
    res = alifestd_mark_lineage_affine_asexual(df, "m", "a")
    assert np.isnan(res["lineage_affine"].to_numpy()[[1, 2]]).all()
    assert not np.isnan(res["lineage_affine"].to_numpy()[[0, 3, 4]]).any()


def test_matches_cumsum_and_cumprod():
    # 0 <- 1 <- 2, 3; 0 <- 4 <- 5
    df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3, 4, 5],
            "ancestor_id": [0, 0, 1, 1, 0, 4],
            "v": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "one": 1.0,
        },
    )
    df["v_root"] = np.where(df["id"] == df["ancestor_id"], df["v"], 0.0)

    cumsum = alifestd_mark_lineage_cumsum_asexual(df, "v")["lineage_cumsum"]
    affine = alifestd_mark_lineage_affine_asexual(df, "one", "v")
    np.testing.assert_allclose(affine["lineage_affine"], cumsum)

    cumprod = alifestd_mark_lineage_cumprod_asexual(df, "v")["lineage_cumprod"]
    affine = alifestd_mark_lineage_affine_asexual(df, "v", "v_root")
    np.testing.assert_allclose(affine["lineage_affine"], cumprod)


def test_empty():
    df = pd.DataFrame(
        {"id": [], "ancestor_id": [], "m": [], "a": []}, dtype=float
    )
    res = alifestd_mark_lineage_affine_asexual(df, "m", "a")
    assert "lineage_affine" in res.columns
    assert len(res) == 0


def test_missing_column():
    with pytest.raises(ValueError):
        alifestd_mark_lineage_affine_asexual(_make_df(), "m", "nope")


def test_non_contiguous_ids():
    df = pd.DataFrame(
        {"id": [0, 2], "ancestor_id": [0, 0], "m": [1.0, 1.0], "a": [1.0, 1.0]}
    )
    with pytest.raises(NotImplementedError):
        alifestd_mark_lineage_affine_asexual(df, "m", "a")


def test_non_topologically_sorted():
    df = pd.DataFrame(
        {"id": [0, 1], "ancestor_id": [1, 1], "m": [1.0, 1.0], "a": [1.0, 1.0]}
    )
    with pytest.raises(NotImplementedError):
        alifestd_mark_lineage_affine_asexual(df, "m", "a")
