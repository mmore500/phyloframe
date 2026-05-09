import typing

import polars as pl
import pytest

from phyloframe.legacy import alifestd_mark_lineage_cumprod_polars


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
                "id": [0, 1, 2, 3],
                "ancestor_id": [0, 0, 1, 2],
                "v": [2.0, 3.0, 5.0, 7.0],
            }
        ),
    )
    res = alifestd_mark_lineage_cumprod_polars(df_pl, "v").lazy().collect()
    assert res["lineage_cumprod"].to_list() == [2.0, 6.0, 30.0, 210.0]


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
                "id": [0, 1, 2, 3],
                "ancestor_id": [0, 0, 0, 0],
                "v": [2.0, 3.0, 5.0, 7.0],
            }
        ),
    )
    res = (
        alifestd_mark_lineage_cumprod_polars(df_pl, "v", reverse=True)
        .lazy()
        .collect()
    )
    assert res["lineage_cumprod"].to_list() == [210.0, 3.0, 5.0, 7.0]


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
    res = alifestd_mark_lineage_cumprod_polars(df_pl, "v").lazy().collect()
    assert res["lineage_cumprod"].to_list() == [9.0]


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
    res = alifestd_mark_lineage_cumprod_polars(df_pl, "v").lazy().collect()
    assert "lineage_cumprod" in res.columns
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
                "v": [2.0, 3.0, 4.0, 5.0],
            }
        ),
    )
    res = alifestd_mark_lineage_cumprod_polars(df_pl, "v").lazy().collect()
    assert res["lineage_cumprod"].to_list() == [2.0, 3.0, 8.0, 15.0]


def test_missing_values_column():
    df_pl = pl.DataFrame(
        {"id": [0, 1], "ancestor_id": [0, 0], "v": [1.0, 2.0]}
    )
    with pytest.raises(ValueError):
        alifestd_mark_lineage_cumprod_polars(df_pl, "no_such_col")
