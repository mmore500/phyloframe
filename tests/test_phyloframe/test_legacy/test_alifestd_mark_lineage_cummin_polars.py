import typing

import numpy as np
import polars as pl
import pytest

from phyloframe.legacy import alifestd_mark_lineage_cummin_polars


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_simple_tree(apply: typing.Callable):
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0, 1, 2, 3, 4],
                "ancestor_id": [0, 0, 0, 1, 1],
                "v": [10.0, 5.0, 20.0, 8.0, 1.0],
            }
        ),
    )
    res = alifestd_mark_lineage_cummin_polars(df_pl, "v").lazy().collect()
    assert res["lineage_cummin"].to_list() == [10.0, 5.0, 10.0, 5.0, 1.0]


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_reverse(apply: typing.Callable):
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0, 1, 2, 3, 4],
                "ancestor_id": [0, 0, 0, 1, 1],
                "v": [10.0, 5.0, 20.0, 8.0, 1.0],
            }
        ),
    )
    res = (
        alifestd_mark_lineage_cummin_polars(df_pl, "v", reverse=True)
        .lazy()
        .collect()
    )
    assert res["lineage_cummin"].to_list() == [1.0, 1.0, 20.0, 8.0, 1.0]


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_singleton(apply: typing.Callable):
    df_pl = apply(
        pl.DataFrame({"id": [0], "ancestor_id": [0], "v": [9.0]}),
    )
    res = alifestd_mark_lineage_cummin_polars(df_pl, "v").lazy().collect()
    assert res["lineage_cummin"].to_list() == [9.0]


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_empty(apply: typing.Callable):
    df_pl = apply(
        pl.DataFrame(
            {"id": [], "ancestor_id": [], "v": []},
            schema={
                "id": pl.Int64,
                "ancestor_id": pl.Int64,
                "v": pl.Float64,
            },
        ),
    )
    res = alifestd_mark_lineage_cummin_polars(df_pl, "v").lazy().collect()
    assert "lineage_cummin" in res.columns
    assert res.is_empty()


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_forest(apply: typing.Callable):
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0, 1, 2, 3],
                "ancestor_id": [0, 1, 0, 1],
                "v": [10.0, 20.0, 5.0, 30.0],
            }
        ),
    )
    res = alifestd_mark_lineage_cummin_polars(df_pl, "v").lazy().collect()
    assert res["lineage_cummin"].to_list() == [10.0, 20.0, 5.0, 20.0]


def test_missing_values_column():
    df_pl = pl.DataFrame(
        {"id": [0, 1], "ancestor_id": [0, 0], "v": [1.0, 2.0]}
    )
    with pytest.raises(ValueError):
        alifestd_mark_lineage_cummin_polars(df_pl, "no_such_col")


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
@pytest.mark.parametrize("missing", [None, float("nan")])
@pytest.mark.parametrize("reverse", [False, True])
def test_skipna_no_value_in_scope(
    apply: typing.Callable, missing: typing.Optional[float], reverse: bool
):
    # 0 <- 1 <- 2, 0 <- 3
    df = pl.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "ancestor_id": [0, 0, 1, 0],
            "v": pl.Series([missing, missing, 2.0, -np.inf], dtype=pl.Float64),
        }
    )
    res = (
        alifestd_mark_lineage_cummin_polars(apply(df), "v", reverse=reverse)
        .lazy()
        .collect()
    )
    # null where no non-missing value in scope; genuine inf values retained
    expected = (
        [-np.inf, 2.0, 2.0, -np.inf] if reverse else [None, None, 2.0, -np.inf]
    )
    assert res["lineage_cummin"].to_list() == expected


def test_skipna_int_with_nulls():
    df = pl.DataFrame(
        {
            "id": [0, 1, 2],
            "ancestor_id": [0, 0, 1],
            "v": pl.Series([None, 3, None], dtype=pl.Int64),
        }
    )
    res = alifestd_mark_lineage_cummin_polars(df, "v")
    assert res["lineage_cummin"].to_list() == [None, 3.0, 3.0]
