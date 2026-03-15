import os
import typing

import pandas as pd
import polars as pl
import pytest

from phyloframe.legacy import alifestd_has_compact_ids
from phyloframe.legacy._alifestd_has_compact_ids_polars import (
    alifestd_has_compact_ids_polars as alifestd_has_compact_ids_polars_,
)

from ._impl import enforce_dtype_stability_polars

alifestd_has_compact_ids_polars = enforce_dtype_stability_polars(
    alifestd_has_compact_ids_polars_
)

assets_path = os.path.join(os.path.dirname(__file__), "assets")


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_empty(apply: typing.Callable):
    df = apply(pl.DataFrame({"id": []}, schema={"id": pl.Int64}))
    assert alifestd_has_compact_ids_polars(df)


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_single_zero(apply: typing.Callable):
    df = apply(pl.DataFrame({"id": [0]}))
    assert alifestd_has_compact_ids_polars(df)


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_single_nonzero(apply: typing.Callable):
    df = apply(pl.DataFrame({"id": [5]}))
    assert not alifestd_has_compact_ids_polars(df)


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_contiguous(apply: typing.Callable):
    df = apply(pl.DataFrame({"id": [0, 1, 2]}))
    assert alifestd_has_compact_ids_polars(df)


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_compact_unordered(apply: typing.Callable):
    """Ids 2, 0, 1 are compact (max==2, len==3)."""
    df = apply(pl.DataFrame({"id": [2, 0, 1]}))
    assert alifestd_has_compact_ids_polars(df)


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_not_compact_gap(apply: typing.Callable):
    """Ids 0, 2, 4 have gaps and max != len-1."""
    df = apply(pl.DataFrame({"id": [0, 2, 4]}))
    assert not alifestd_has_compact_ids_polars(df)


@pytest.mark.parametrize(
    "phylogeny_df",
    [
        pd.read_csv(
            f"{assets_path}/example-standard-toy-sexual-phylogeny.csv"
        ),
        pd.read_csv(
            f"{assets_path}/example-standard-toy-asexual-phylogeny.csv"
        ),
        pd.read_csv(f"{assets_path}/nk_ecoeaselection.csv"),
        pd.read_csv(f"{assets_path}/nk_lexicaseselection.csv"),
        pd.read_csv(f"{assets_path}/nk_tournamentselection.csv"),
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
    result_pd = alifestd_has_compact_ids(phylogeny_df)
    df = apply(pl.from_pandas(phylogeny_df))
    result_pl = alifestd_has_compact_ids_polars(df)
    assert result_pd == result_pl
