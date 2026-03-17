import os

import pandas as pd
import pytest

from phyloframe.legacy import (
    alifestd_mark_sample_tips_lineage_stratified_asexual,
    alifestd_to_working_format,
)

assets_path = os.path.join(os.path.dirname(__file__), "assets")


@pytest.mark.parametrize("n_downsample", [None, 4])
def test_alifestd_mark_sample_tips_lineage_stratified_asexual(n_downsample):
    phylogeny_df = alifestd_to_working_format(
        pd.read_csv(f"{assets_path}/nk_ecoeaselection.csv")
    )
    original_df = phylogeny_df.copy()

    result = alifestd_mark_sample_tips_lineage_stratified_asexual(
        phylogeny_df, n_downsample=n_downsample, seed=1
    )

    col = "alifestd_mark_sample_tips_lineage_stratified_asexual"
    assert col in result.columns
    assert result[col].dtype == bool
    assert len(result) == len(original_df)
    pd.testing.assert_frame_equal(phylogeny_df, original_df)


def test_alifestd_mark_sample_tips_lineage_stratified_asexual_mark_as():
    phylogeny_df = alifestd_to_working_format(
        pd.read_csv(f"{assets_path}/nk_ecoeaselection.csv")
    )

    result = alifestd_mark_sample_tips_lineage_stratified_asexual(
        phylogeny_df, n_downsample=4, seed=1, mark_as="my_col"
    )

    assert "my_col" in result.columns
    assert result["my_col"].dtype == bool


def test_alifestd_mark_sample_tips_lineage_stratified_asexual_empty():
    phylogeny_df = pd.DataFrame(
        {"id": [], "parent_id": [], "ancestor_id": [], "origin_time": []}
    )

    col = "alifestd_mark_sample_tips_lineage_stratified_asexual"
    result = alifestd_mark_sample_tips_lineage_stratified_asexual(
        phylogeny_df,
    )

    assert result.empty
    assert col in result.columns
