import os

import pandas as pd
import polars as pl
import pytest

from phyloframe.legacy import alifestd_to_working_format
from phyloframe.legacy._alifestd_mark_sample_tips_lineage_stratified_polars import (
    alifestd_mark_sample_tips_lineage_stratified_polars,
)

assets_path = os.path.join(os.path.dirname(__file__), "assets")


@pytest.mark.parametrize("n_downsample", [None, 4])
def test_alifestd_mark_sample_tips_lineage_stratified_polars(n_downsample):
    phylogeny_df = pl.from_pandas(
        alifestd_to_working_format(
            pd.read_csv(f"{assets_path}/nk_ecoeaselection.csv")
        )
    )
    n_rows = len(phylogeny_df)

    result = alifestd_mark_sample_tips_lineage_stratified_polars(
        phylogeny_df, n_downsample=n_downsample, seed=1
    )

    col = "alifestd_mark_sample_tips_lineage_stratified_polars"
    assert col in result.columns
    assert result[col].dtype == pl.Boolean
    assert len(result) == n_rows


def test_alifestd_mark_sample_tips_lineage_stratified_polars_mark_as():
    phylogeny_df = pl.from_pandas(
        alifestd_to_working_format(
            pd.read_csv(f"{assets_path}/nk_ecoeaselection.csv")
        )
    )

    result = alifestd_mark_sample_tips_lineage_stratified_polars(
        phylogeny_df, n_downsample=4, seed=1, mark_as="my_col"
    )

    assert "my_col" in result.columns
