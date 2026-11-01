import numpy as np
import pandas as pd
import pytest

from phyloframe._auxlib import identity
from phyloframe.legacy import alifestd_tgs_allmdc_estimate_asexual
from phyloframe.legacy._impl import tgs_allmdc_impl, tgs_prepare_asexual

from ..test_alifestd_tgs_allmdc_estimate_asexual import _make_cherries

_KWARGS = dict(
    calibrate=True,
    calibration_neighborhood_fractions=(0.25, 1.0),
    num_calibration_partitions=64,
    clade_size_calibration_ratio=2.0,
    min_calibration_block_neutral_gains=1,
    num_survival_time_bins=16,
    num_bootstrap=2,
    loss_rate=0.0,
    seed=1,
    parallel_backend=None,
)


def test_matches_public():
    phylogeny_df, traits = _make_cherries(2_000)
    names = ["focal", "n1", "n2", "n3", "n4"]
    prepared = tgs_prepare_asexual(
        phylogeny_df, False, trait_masks=traits, trait_names=names
    )
    res = tgs_allmdc_impl(
        prepared,
        trait_names=names,
        neutral_traits=[1, 2, 3, 4],
        selection_starts=(0.0, -0.3, -1.8),
        max_growth_exponent=600.0,
        progress_wrap=identity,
        **_KWARGS,
    )
    expected = alifestd_tgs_allmdc_estimate_asexual(
        phylogeny_df,
        trait_masks=traits,
        trait_names=names,
        neutral_traits=[1, 2, 3, 4],
        **_KWARGS,
    )
    pd.testing.assert_frame_equal(res, expected)
    assert "selection_se" in res.columns


def test_no_branches():
    prepared = pd.DataFrame(
        {
            "ancestor_id": [0],
            "origin_time": [0.0],
            "__alifestd_tgs_branch_length": [np.nan],
            "__alifestd_tgs_trait0": [False],
        },
    )
    res = tgs_allmdc_impl(
        prepared,
        trait_names=["t"],
        neutral_traits=[],
        selection_starts=(0.0,),
        max_growth_exponent=600.0,
        progress_wrap=identity,
        **{**_KWARGS, "calibrate": False},
    )
    assert res["num_leaves"].tolist() == [0]
    assert "selection_p" in res.columns


def test_num_survival_time_bins_validated():
    prepared = pd.DataFrame(
        {
            "ancestor_id": [0],
            "origin_time": [0.0],
            "__alifestd_tgs_branch_length": [np.nan],
            "__alifestd_tgs_trait0": [False],
        },
    )
    with pytest.raises(ValueError):
        tgs_allmdc_impl(
            prepared,
            trait_names=["t"],
            neutral_traits=[],
            selection_starts=(0.0,),
            max_growth_exponent=600.0,
            progress_wrap=identity,
            **{**_KWARGS, "num_survival_time_bins": 0},
        )
