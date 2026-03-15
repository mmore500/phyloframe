import os
import typing

import pandas as pd
import polars as pl
import pytest

from phyloframe.legacy import alifestd_is_asexual
from phyloframe.legacy._alifestd_is_asexual_polars import (
    alifestd_is_asexual_polars as alifestd_is_asexual_polars_,
)

from ._impl import enforce_dtype_stability_polars

alifestd_is_asexual_polars = enforce_dtype_stability_polars(
    alifestd_is_asexual_polars_
)

assets_path = os.path.join(os.path.dirname(__file__), "assets")


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_with_ancestor_id(apply: typing.Callable):
    """If ancestor_id column exists, always True."""
    df = apply(
        pl.DataFrame(
            {
                "id": [0, 1],
                "ancestor_id": [0, 0],
                "ancestor_list": ["[none]", "[0]"],
            }
        )
    )
    assert alifestd_is_asexual_polars(df)


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_asexual_no_ancestor_id(apply: typing.Callable):
    df = apply(
        pl.DataFrame(
            {"id": [0, 1, 2], "ancestor_list": ["[none]", "[0]", "[0]"]}
        )
    )
    assert alifestd_is_asexual_polars(df)


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_sexual_no_ancestor_id(apply: typing.Callable):
    df = apply(
        pl.DataFrame(
            {"id": [0, 1, 2], "ancestor_list": ["[none]", "[none]", "[0,1]"]}
        )
    )
    assert not alifestd_is_asexual_polars(df)


@pytest.mark.parametrize(
    "phylogeny_df",
    [
        pd.read_csv(
            f"{assets_path}/example-standard-toy-sexual-phylogeny.csv"
        ),
        pd.read_csv(
            f"{assets_path}/example-standard-toy-asexual-phylogeny.csv"
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
    result_pd = alifestd_is_asexual(phylogeny_df)
    df = apply(pl.from_pandas(phylogeny_df))
    result_pl = alifestd_is_asexual_polars(df)
    assert result_pd == result_pl
