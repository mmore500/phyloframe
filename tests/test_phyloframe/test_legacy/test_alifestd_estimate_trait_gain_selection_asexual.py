import numpy as np
import pandas as pd
import pytest

from phyloframe.legacy import alifestd_estimate_trait_gain_selection_asexual


def _make_cherries(num_cherries: int = 40_000, seed: int = 1):
    """Root with cherries hanging off it; each leaf's trait is gained on its
    pendant branch under the exposure model.

    Trait 0 has twice the neutral gain rate and a cost; traits 1-4 are
    neutral.
    """
    rng = np.random.default_rng(seed)
    num_nodes = 1 + 3 * num_cherries
    inner = np.arange(1, num_nodes, 3)
    ancestor_id = np.zeros(num_nodes, dtype=int)
    ancestor_id[inner + 1] = ancestor_id[inner + 2] = inner
    origin_time = np.zeros(num_nodes)
    origin_time[inner] = rng.uniform(0, 20, num_cherries)
    leaves = np.concatenate([inner + 1, inner + 2])
    length = rng.uniform(0.5, 60, len(leaves))
    origin_time[leaves] = origin_time[ancestor_id[leaves]] + length

    traits = np.zeros((5, num_nodes), dtype=bool)
    for trait, (gain, selection) in enumerate(
        [(2e-3, -0.05)] + [(1e-3, 0.0)] * 4
    ):
        p = (
            gain * length
            if selection == 0
            else gain * np.expm1(selection * length) / selection
        )
        traits[trait, leaves] = rng.random(len(leaves)) < p

    phylogeny_df = pd.DataFrame(
        {
            "id": range(num_nodes),
            "ancestor_id": ancestor_id,
            "origin_time": origin_time,
        },
    )
    blocks = np.arange(num_nodes) // 3 % 50
    return phylogeny_df, traits, blocks


@pytest.mark.parametrize("calibrate", [False, True])
def test_recovers_gain_selection(calibrate: bool):
    phylogeny_df, traits, blocks = _make_cherries()
    original_df = phylogeny_df.copy()
    res = alifestd_estimate_trait_gain_selection_asexual(
        phylogeny_df,
        trait_masks=traits,
        trait_names=["focal", "n1", "n2", "n3", "n4"],
        neutral_traits=[1, 2, 3, 4],
        calibrate=calibrate,
        block_ids=blocks,
        num_bootstrap=20,
        seed=1,
    )
    assert original_df.equals(phylogeny_df)
    assert res["trait"].tolist() == ["focal", "n1", "n2", "n3", "n4"]
    assert (res["num_leaves"] <= 80_000).all()

    focal = res.iloc[0]
    assert focal["gain_ratio"] == pytest.approx(2.0, rel=0.2)
    assert focal["selection"] == pytest.approx(-0.05, abs=0.02)
    assert focal["selection_p"] < 1e-3
    assert focal["gain_ratio_p"] < 1e-3

    neutral = res.iloc[1:]
    # gain rate trades off against selection, so with ~2k carriers per
    # trait, neutral gain ratios scatter by ~20%
    assert neutral["gain_ratio"].tolist() == pytest.approx([1.0] * 4, rel=0.3)
    assert (neutral["selection"].abs() < 0.02).all()
    assert np.isfinite(res["selection_se"]).all()
    assert np.isfinite(res["log_gain_ratio_se"]).all()


def test_row_order_invariant():
    phylogeny_df, traits, blocks = _make_cherries(2_000)
    kwargs = dict(neutral_traits=[1, 2, 3, 4], calibrate=True)
    expected = alifestd_estimate_trait_gain_selection_asexual(
        phylogeny_df, trait_masks=traits, block_ids=blocks, **kwargs
    )

    order = np.random.default_rng(2).permutation(len(phylogeny_df))
    shuffled = phylogeny_df.iloc[order].reset_index(drop=True)
    shuffled["id"] += 7
    shuffled["ancestor_id"] += 7
    res = alifestd_estimate_trait_gain_selection_asexual(
        shuffled,
        trait_masks=traits[:, order],
        block_ids=blocks[order],
        **kwargs,
    )
    pd.testing.assert_frame_equal(res, expected, rtol=1e-6)


def test_drops_inherited_and_rootward():
    #     0
    #     1
    #   2   3
    phylogeny_df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "ancestor_id": [0, 0, 1, 1],
            "origin_time": [0.0, 1.0, 2.0, 4.0],
        },
    )
    res = alifestd_estimate_trait_gain_selection_asexual(
        phylogeny_df,
        trait_masks=[[0, 0, 1, 1], [0, 0, 1, 0]],
        min_leaf_origin_time=0.0,
        num_exposure_bins=4,
    )
    # both leaves carry trait 0 from above their parent, so both drop
    assert res["num_leaves"].tolist() == [0, 2]
    assert res["num_trait_present"].tolist() == [0, 1]
    assert np.isnan(res.loc[0, "selection"])
    assert np.isnan(res["gain_ratio"]).all()

    res = alifestd_estimate_trait_gain_selection_asexual(
        phylogeny_df,
        trait_masks=[[0, 0, 1, 1], [0, 0, 1, 0]],
        min_leaf_origin_time=3.0,
        num_exposure_bins=4,
    )
    assert res["num_leaves"].tolist() == [0, 1]


def test_calibrate_requires_neutral():
    phylogeny_df, traits, __ = _make_cherries(100)
    with pytest.raises(ValueError):
        alifestd_estimate_trait_gain_selection_asexual(
            phylogeny_df,
            trait_masks=traits,
            calibrate=True,
            neutral_traits=[1],
        )
