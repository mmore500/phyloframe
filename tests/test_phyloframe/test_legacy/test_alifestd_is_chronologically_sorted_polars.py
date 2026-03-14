import os
import typing

import pandas as pd
import polars as pl
import pytest

from phyloframe.legacy import alifestd_is_chronologically_sorted
from phyloframe.legacy._alifestd_is_chronologically_sorted_polars import (
    alifestd_is_chronologically_sorted_polars as alifestd_is_chronologically_sorted_polars_,
)

from ._impl import enforce_dtype_stability_polars

alifestd_is_chronologically_sorted_polars = enforce_dtype_stability_polars(
    alifestd_is_chronologically_sorted_polars_
)

assets_path = os.path.join(os.path.dirname(__file__), "assets")


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_sorted(apply: typing.Callable):
    df = apply(pl.DataFrame({"id": [0, 1, 2], "origin_time": [1.0, 2.0, 3.0]}))
    assert alifestd_is_chronologically_sorted_polars(df)


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_not_sorted(apply: typing.Callable):
    df = apply(pl.DataFrame({"id": [0, 1, 2], "origin_time": [3.0, 1.0, 2.0]}))
    assert not alifestd_is_chronologically_sorted_polars(df)


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_equal_times(apply: typing.Callable):
    df = apply(pl.DataFrame({"id": [0, 1, 2], "origin_time": [1.0, 1.0, 1.0]}))
    assert alifestd_is_chronologically_sorted_polars(df)


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_no_origin_time_column(apply: typing.Callable):
    df = apply(pl.DataFrame({"id": [0, 1, 2]}))
    assert not alifestd_is_chronologically_sorted_polars(df)


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_single_row(apply: typing.Callable):
    df = apply(pl.DataFrame({"id": [0], "origin_time": [5.0]}))
    assert alifestd_is_chronologically_sorted_polars(df)


@pytest.mark.parametrize(
    "phylogeny_df",
    [
        pd.read_csv(
            f"{assets_path}/example-standard-toy-asexual-bifurcating-phylogeny.csv"
        ),
    ],
)
@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_matches_pandas(phylogeny_df: pd.DataFrame, apply: typing.Callable):
    result_pd = alifestd_is_chronologically_sorted(phylogeny_df)
    df = apply(pl.from_pandas(phylogeny_df))
    result_pl = alifestd_is_chronologically_sorted_polars(df)
    assert result_pd == result_pl
