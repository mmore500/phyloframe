import typing

import numpy as np
import pandas as pd
import polars as pl
import pytest

from phyloframe.legacy import (
    alifestd_find_oldest_root_id_asexual,
    alifestd_find_oldest_root_id_polars,
)

apply_params = pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)


@apply_params
def test_empty(apply: typing.Callable):
    df_pl = pl.DataFrame(
        {
            "id": pl.Series([], dtype=pl.Int64),
            "ancestor_id": pl.Series([], dtype=pl.Int64),
        },
    )
    assert alifestd_find_oldest_root_id_polars(apply(df_pl)) is None


@apply_params
def test_lowest_id_without_origin_time(apply: typing.Callable):
    df_pl = pl.DataFrame({"id": [0, 1, 2, 3], "ancestor_id": [1, 1, 3, 3]})
    assert alifestd_find_oldest_root_id_polars(apply(df_pl)) == 1


@apply_params
def test_lowest_origin_time_ties_by_id(apply: typing.Callable):
    df_pl = pl.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "ancestor_id": [0, 1, 2, 0],
            "origin_time": [3.0, 1.0, 1.0, 0.0],
        },
    )
    assert alifestd_find_oldest_root_id_polars(apply(df_pl)) == 1


def test_integer_origin_time():
    df_pl = pl.DataFrame(
        {"id": [0, 1], "ancestor_id": [0, 1], "origin_time": [5, 2]},
    )
    assert alifestd_find_oldest_root_id_polars(df_pl) == 1


@pytest.mark.parametrize("bad", [None, float("nan")], ids=["null", "nan"])
def test_null_or_nan_root_origin_time_raises(bad):
    df_pl = pl.DataFrame(
        {"id": [0, 1], "ancestor_id": [0, 1]},
    ).with_columns(origin_time=pl.Series([bad, 1.0], dtype=pl.Float64))
    with pytest.raises(NotImplementedError, match="origin_time"):
        alifestd_find_oldest_root_id_polars(df_pl)


@pytest.mark.parametrize("seed", range(3))
def test_matches_pandas(seed: int):
    rng = np.random.default_rng(seed)
    n = 50
    ancestor_ids = np.array(
        [i if rng.random() < 0.3 else rng.integers(i + 1) for i in range(n)]
    )
    df_pd = pd.DataFrame(
        {
            "id": np.arange(n),
            "ancestor_id": ancestor_ids,
            "origin_time": rng.integers(0, 5, n).astype(float),
        },
    )
    assert alifestd_find_oldest_root_id_polars(
        pl.from_pandas(df_pd)
    ) == alifestd_find_oldest_root_id_asexual(df_pd)
