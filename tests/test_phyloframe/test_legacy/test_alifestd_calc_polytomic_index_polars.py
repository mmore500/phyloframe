import os
import typing

import pandas as pd
import polars as pl
import pytest

from phyloframe.legacy import alifestd_calc_polytomic_index
from phyloframe.legacy._alifestd_calc_polytomic_index_polars import (
    alifestd_calc_polytomic_index_polars as alifestd_calc_polytomic_index_polars_,
)

from ._impl import enforce_dtype_stability_polars

alifestd_calc_polytomic_index_polars = enforce_dtype_stability_polars(
    alifestd_calc_polytomic_index_polars_
)

assets_path = os.path.join(os.path.dirname(__file__), "assets")


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_bifurcating(apply: typing.Callable):
    """A strictly bifurcating tree has polytomic index 0."""
    df = apply(
        pl.DataFrame(
            {
                "id": [0, 1, 2, 3, 4],
                "ancestor_id": [0, 0, 0, 1, 1],
            }
        )
    )
    assert alifestd_calc_polytomic_index_polars(df) == 0


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
    result_pd = alifestd_calc_polytomic_index(phylogeny_df)
    df = apply(pl.from_pandas(phylogeny_df))
    result_pl = alifestd_calc_polytomic_index_polars(df)
    assert result_pd == result_pl
