import typing

import numpy as np
import polars as pl
import pytest

from phyloframe.legacy import (
    alifestd_mark_lineage_delta_asexual,
    alifestd_mark_lineage_delta_polars,
)

from .test_alifestd_mark_lineage_delta_asexual import _make_tree

apply_params = pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)


@apply_params
def test_matches_pandas(apply: typing.Callable):
    df_pd = _make_tree().assign(trait=[False, True, True, False, False])
    for values in ("v", "trait"):
        res = (
            alifestd_mark_lineage_delta_polars(
                apply(pl.from_pandas(df_pd)), values
            )
            .lazy()
            .collect()
        )
        expected = alifestd_mark_lineage_delta_asexual(df_pd, values)
        assert (
            res["lineage_delta"].to_list()
            == expected["lineage_delta"].tolist()
        )


def test_expression_values():
    df_pl = pl.from_pandas(_make_tree())
    res = alifestd_mark_lineage_delta_polars(
        df_pl, pl.col("v") * 2, mark_as="d"
    )
    assert res["d"].to_list() == [10, 4, -6, 0, 8]


@apply_params
def test_empty(apply: typing.Callable):
    df_pl = pl.DataFrame(
        {
            "id": pl.Series([], dtype=pl.Int64),
            "ancestor_id": pl.Series([], dtype=pl.Int64),
            "v": pl.Series([], dtype=pl.Int64),
        },
    )
    res = alifestd_mark_lineage_delta_polars(apply(df_pl), "v")
    res = res.lazy().collect()
    assert res.is_empty()
    assert "lineage_delta" in res.columns


def test_missing_values_column_raises():
    with pytest.raises(ValueError):
        alifestd_mark_lineage_delta_polars(
            pl.from_pandas(_make_tree()), "absent"
        )


def test_non_contiguous_raises():
    df_pl = pl.from_pandas(_make_tree()).with_columns(
        id=pl.Series([0, 1, 2, 3, 5])
    )
    with pytest.raises(NotImplementedError):
        alifestd_mark_lineage_delta_polars(df_pl, "v")


def test_keeps_input_dtype():
    df_pl = pl.from_pandas(
        _make_tree().assign(v=np.array([5, 7, 4, 7, 9], dtype=np.uint8))
    )
    res = alifestd_mark_lineage_delta_polars(df_pl, "v")
    # negative deltas of unsigned values wrap around
    assert res["lineage_delta"].to_numpy().tolist() == [5, 2, 253, 0, 4]
    assert res["lineage_delta"].to_numpy().dtype == np.uint8
