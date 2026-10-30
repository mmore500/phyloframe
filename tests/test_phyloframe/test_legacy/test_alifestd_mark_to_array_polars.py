import typing

import numpy as np
import polars as pl
import pytest

from phyloframe.legacy import (
    alifestd_mark_lineage_affine_polars,
    alifestd_mark_lineage_cumsum_polars,
    alifestd_mark_num_children_polars,
    alifestd_mark_to_array_polars,
)


def _make_df() -> pl.DataFrame:
    # 0 <- 1 <- 2, 0 <- 3
    return pl.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "ancestor_id": [0, 0, 1, 0],
            "v": [1, 2, 3, 4],
            "m": [9.0, 2.0, 3.0, 0.5],
        },
    )


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_single_column(apply: typing.Callable):
    res = alifestd_mark_to_array_polars(
        apply(_make_df()), alifestd_mark_lineage_cumsum_polars, values="v"
    )
    assert isinstance(res, np.ndarray)
    np.testing.assert_array_equal(res, [1, 3, 6, 5])


def test_multiple_columns():
    res = alifestd_mark_to_array_polars(
        _make_df(), alifestd_mark_lineage_affine_polars, scale="m", offset="v"
    )
    np.testing.assert_allclose(res, [1.0, 4.0, 15.0, 4.5])


def test_no_columns():
    res = alifestd_mark_to_array_polars(
        _make_df(), alifestd_mark_num_children_polars
    )
    np.testing.assert_array_equal(res, [2, 1, 0, 0])


def test_kwargs_forwarded():
    res = alifestd_mark_to_array_polars(
        _make_df(),
        alifestd_mark_lineage_cumsum_polars,
        values="v",
        reverse=True,
    )
    np.testing.assert_array_equal(res, [10, 5, 3, 4])


def test_input_unchanged():
    df = _make_df()
    alifestd_mark_to_array_polars(
        df, alifestd_mark_lineage_cumsum_polars, values="v"
    )
    assert df.columns == ["id", "ancestor_id", "v", "m"]


def test_non_contiguous_ids_not_implemented():
    df = pl.DataFrame({"id": [0, 2], "ancestor_id": [0, 0], "v": [1, 2]})
    with pytest.raises(NotImplementedError):
        alifestd_mark_to_array_polars(
            df, alifestd_mark_lineage_cumsum_polars, values="v"
        )


def test_non_topologically_sorted_not_implemented():
    df = pl.DataFrame({"id": [0, 1], "ancestor_id": [1, 1], "v": [1, 2]})
    with pytest.raises(NotImplementedError):
        alifestd_mark_to_array_polars(
            df, alifestd_mark_lineage_cumsum_polars, values="v"
        )


def test_reordering_op_raises():
    def reordering_op(phylogeny_df, mark_as):
        return phylogeny_df.with_columns(pl.col("v").alias(mark_as)).reverse()

    with pytest.raises(AssertionError):
        alifestd_mark_to_array_polars(_make_df(), reordering_op)
