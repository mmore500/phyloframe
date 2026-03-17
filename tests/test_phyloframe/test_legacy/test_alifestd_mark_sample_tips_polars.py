import os

import pandas as pd
import polars as pl
import pytest

from phyloframe.legacy import alifestd_to_working_format
from phyloframe.legacy._alifestd_mark_sample_tips_polars import (
    alifestd_mark_sample_tips_polars,
)

assets_path = os.path.join(os.path.dirname(__file__), "assets")


@pytest.mark.parametrize("seed", [1, 42])
@pytest.mark.parametrize("n_sample", [1, 5, 100000000])
def test_alifestd_mark_sample_tips_polars(seed, n_sample):
    phylogeny_df = pl.from_pandas(
        alifestd_to_working_format(
            pd.read_csv(f"{assets_path}/nk_ecoeaselection.csv")
        )
    )
    n_rows = len(phylogeny_df)

    result = alifestd_mark_sample_tips_polars(
        phylogeny_df, n_sample, seed=seed
    )

    assert "alifestd_mark_sample_tips_polars" in result.columns
    assert result["alifestd_mark_sample_tips_polars"].dtype == pl.Boolean
    assert len(result) == n_rows


def test_alifestd_mark_sample_tips_polars_mark_as():
    phylogeny_df = pl.from_pandas(
        alifestd_to_working_format(
            pd.read_csv(f"{assets_path}/nk_ecoeaselection.csv")
        )
    )

    result = alifestd_mark_sample_tips_polars(
        phylogeny_df, 5, seed=1, mark_as="my_col"
    )

    assert "my_col" in result.columns


def test_alifestd_mark_sample_tips_polars_empty():
    phylogeny_df = pl.DataFrame(
        {
            "id": pl.Series([], dtype=pl.Int64),
            "ancestor_id": pl.Series([], dtype=pl.Int64),
        }
    )

    result = alifestd_mark_sample_tips_polars(phylogeny_df, 1)

    assert result.is_empty()
    assert "alifestd_mark_sample_tips_polars" in result.columns
