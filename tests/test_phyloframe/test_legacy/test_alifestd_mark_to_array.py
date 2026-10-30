import numpy as np
import pandas as pd

from phyloframe.legacy import (
    alifestd_mark_lineage_affine_asexual,
    alifestd_mark_lineage_cumsum_asexual,
    alifestd_mark_num_children_asexual,
    alifestd_mark_to_array,
)


def _make_df() -> pd.DataFrame:
    # 0 <- 1 <- 2, 0 <- 3
    return pd.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "ancestor_id": [0, 0, 1, 0],
            "v": [1, 2, 3, 4],
            "m": [9.0, 2.0, 3.0, 0.5],
        },
    )


def test_single_column():
    df = _make_df()
    original_df = df.copy()
    res = alifestd_mark_to_array(df, alifestd_mark_lineage_cumsum_asexual, "v")
    assert isinstance(res, np.ndarray)
    np.testing.assert_array_equal(res, [1, 3, 6, 5])
    pd.testing.assert_frame_equal(df, original_df)


def test_multiple_columns():
    res = alifestd_mark_to_array(
        _make_df(), alifestd_mark_lineage_affine_asexual, "m", "v"
    )
    np.testing.assert_allclose(res, [1.0, 4.0, 15.0, 4.5])


def test_no_columns():
    res = alifestd_mark_to_array(
        _make_df(), alifestd_mark_num_children_asexual
    )
    np.testing.assert_array_equal(res, [2, 1, 0, 0])


def test_kwargs_forwarded():
    res = alifestd_mark_to_array(
        _make_df(),
        alifestd_mark_lineage_cumsum_asexual,
        "v",
        reverse=True,
    )
    np.testing.assert_array_equal(res, [10, 5, 3, 4])


def test_empty():
    df = pd.DataFrame({"id": [], "ancestor_id": [], "v": []}, dtype=int)
    res = alifestd_mark_to_array(df, alifestd_mark_lineage_cumsum_asexual, "v")
    assert len(res) == 0
