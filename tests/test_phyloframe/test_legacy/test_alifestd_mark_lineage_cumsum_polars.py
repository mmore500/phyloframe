import typing

import polars as pl
import pytest

from phyloframe.legacy import (
    alifestd_mark_lineage_cumsum_asexual,
    alifestd_mark_lineage_cumsum_polars,
)


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_simple_tree(apply: typing.Callable):
    """Tree:
    0 (root, v=10)
    +-- 1 (v=1)
    |   +-- 3 (v=3)
    |   +-- 4 (v=4)
    +-- 2 (v=2)
    """
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0, 1, 2, 3, 4],
                "ancestor_id": [0, 0, 0, 1, 1],
                "v": [10.0, 1.0, 2.0, 3.0, 4.0],
            }
        ),
    )
    res = alifestd_mark_lineage_cumsum_polars(df_pl, "v").lazy().collect()
    assert res["lineage_cumsum"].to_list() == [10.0, 11.0, 12.0, 14.0, 15.0]


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
                "v": [10.0, 1.0, 2.0, 3.0, 4.0],
            }
        ),
    )
    res = (
        alifestd_mark_lineage_cumsum_polars(df_pl, "v", reverse=True)
        .lazy()
        .collect()
    )
    assert res["lineage_cumsum"].to_list() == [20.0, 8.0, 2.0, 3.0, 4.0]


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_singleton(apply: typing.Callable):
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0],
                "ancestor_id": [0],
                "v": [42.0],
            }
        ),
    )
    res = alifestd_mark_lineage_cumsum_polars(df_pl, "v").lazy().collect()
    assert res["lineage_cumsum"].to_list() == [42.0]


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
    res = alifestd_mark_lineage_cumsum_polars(df_pl, "v").lazy().collect()
    assert "lineage_cumsum" in res.columns
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
                "v": [10.0, 20.0, 1.0, 2.0],
            }
        ),
    )
    res = alifestd_mark_lineage_cumsum_polars(df_pl, "v").lazy().collect()
    assert res["lineage_cumsum"].to_list() == [10.0, 20.0, 11.0, 22.0]


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_pl_expr(apply: typing.Callable):
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0, 1, 2],
                "ancestor_id": [0, 0, 1],
                "v": [1.0, 2.0, 3.0],
            }
        ),
    )
    res = (
        alifestd_mark_lineage_cumsum_polars(df_pl, pl.col("v") * 2)
        .lazy()
        .collect()
    )
    assert res["lineage_cumsum"].to_list() == [2.0, 6.0, 12.0]


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_non_contiguous_ids(apply: typing.Callable):
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0, 2, 5],
                "ancestor_id": [0, 0, 2],
                "v": [0.0, 1.0, 2.0],
            }
        ),
    )
    with pytest.raises(NotImplementedError):
        alifestd_mark_lineage_cumsum_polars(df_pl, "v").lazy().collect()


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_unsorted(apply: typing.Callable):
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0, 1, 2],
                "ancestor_id": [0, 2, 0],
                "v": [0.0, 1.0, 2.0],
            }
        ),
    )
    with pytest.raises(NotImplementedError):
        alifestd_mark_lineage_cumsum_polars(df_pl, "v").lazy().collect()


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_matches_pandas(apply: typing.Callable):
    import pandas as pd

    df_pd = pd.DataFrame(
        {
            "id": [0, 1, 2, 3, 4, 5],
            "ancestor_list": ["[None]", "[0]", "[0]", "[1]", "[1]", "[2]"],
            "v": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    res_pd = alifestd_mark_lineage_cumsum_asexual(df_pd, "v")

    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0, 1, 2, 3, 4, 5],
                "ancestor_id": [0, 0, 0, 1, 1, 2],
                "v": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            }
        ),
    )
    res_pl = alifestd_mark_lineage_cumsum_polars(df_pl, "v").lazy().collect()
    assert (
        res_pd["lineage_cumsum"].tolist() == res_pl["lineage_cumsum"].to_list()
    )
