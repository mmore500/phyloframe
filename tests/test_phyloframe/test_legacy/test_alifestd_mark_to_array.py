import numpy as np
import pandas as pd
import pytest

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
    res = alifestd_mark_to_array(
        df, alifestd_mark_lineage_cumsum_asexual, values="v"
    )
    assert isinstance(res, np.ndarray)
    np.testing.assert_array_equal(res, [1, 3, 6, 5])
    pd.testing.assert_frame_equal(df, original_df)


def test_multiple_columns():
    res = alifestd_mark_to_array(
        _make_df(), alifestd_mark_lineage_affine_asexual, scale="m", offset="v"
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
        values="v",
        reverse=True,
    )
    np.testing.assert_array_equal(res, [10, 5, 3, 4])


def test_empty():
    df = pd.DataFrame({"id": [], "ancestor_id": [], "v": []}, dtype=int)
    res = alifestd_mark_to_array(
        df, alifestd_mark_lineage_cumsum_asexual, values="v"
    )
    assert len(res) == 0


def test_mutate():
    df = _make_df()
    res = alifestd_mark_to_array(
        df, alifestd_mark_lineage_cumsum_asexual, mutate=True, values="v"
    )
    np.testing.assert_array_equal(res, [1, 3, 6, 5])


def test_non_contiguous_ids_not_implemented():
    df = pd.DataFrame({"id": [0, 2], "ancestor_id": [0, 0], "v": [1, 2]})
    with pytest.raises(NotImplementedError):
        alifestd_mark_to_array(
            df, alifestd_mark_lineage_cumsum_asexual, values="v"
        )


def test_non_topologically_sorted_not_implemented():
    df = pd.DataFrame({"id": [0, 1], "ancestor_id": [1, 1], "v": [1, 2]})
    with pytest.raises(NotImplementedError):
        alifestd_mark_to_array(
            df, alifestd_mark_lineage_cumsum_asexual, values="v"
        )


def test_reordering_op_raises():
    def reordering_op(phylogeny_df, mutate, mark_as):
        phylogeny_df[mark_as] = phylogeny_df["v"]
        return phylogeny_df.iloc[::-1]

    with pytest.raises(AssertionError):
        alifestd_mark_to_array(_make_df(), reordering_op)
