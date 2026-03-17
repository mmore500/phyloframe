import os

import pandas as pd
import pytest

from phyloframe.legacy import alifestd_mark_sample_tips_asexual

assets_path = os.path.join(os.path.dirname(__file__), "assets")


@pytest.mark.parametrize("seed", [1, 42])
@pytest.mark.parametrize("n_sample", [1, 5, 100000000])
def test_alifestd_mark_sample_tips_asexual(seed, n_sample):
    phylogeny_df = pd.read_csv(f"{assets_path}/nk_ecoeaselection.csv")
    original_df = phylogeny_df.copy()

    result = alifestd_mark_sample_tips_asexual(
        phylogeny_df, n_sample, seed=seed
    )

    assert "alifestd_mark_sample_tips_asexual" in result.columns
    assert result["alifestd_mark_sample_tips_asexual"].dtype == bool
    assert len(result) == len(original_df)
    pd.testing.assert_frame_equal(phylogeny_df, original_df)


def test_alifestd_mark_sample_tips_asexual_mark_as():
    phylogeny_df = pd.read_csv(f"{assets_path}/nk_ecoeaselection.csv")

    result = alifestd_mark_sample_tips_asexual(
        phylogeny_df, 5, seed=1, mark_as="my_col"
    )

    assert "my_col" in result.columns
    assert result["my_col"].dtype == bool


def test_alifestd_mark_sample_tips_asexual_empty():
    phylogeny_df = pd.DataFrame({"id": [], "parent_id": [], "ancestor_id": []})

    result = alifestd_mark_sample_tips_asexual(phylogeny_df, 1)

    assert result.empty
    assert "alifestd_mark_sample_tips_asexual" in result.columns
