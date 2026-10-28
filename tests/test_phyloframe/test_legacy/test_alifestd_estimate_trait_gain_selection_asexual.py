import warnings

import numpy as np
import pandas as pd
import pytest

from phyloframe.legacy import alifestd_estimate_trait_gain_selection_asexual
from phyloframe.legacy._alifestd_estimate_trait_gain_selection_asexual import (
    _tgs_bin_leaves,
)


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
    clusters = np.arange(num_nodes) // 3 % 50
    return phylogeny_df, traits, clusters


@pytest.mark.parametrize("calibrate", [False, True])
def test_recovers_gain_selection(calibrate: bool):
    phylogeny_df, traits, clusters = _make_cherries()
    original_df = phylogeny_df.copy()
    res = alifestd_estimate_trait_gain_selection_asexual(
        phylogeny_df,
        trait_masks=traits,
        trait_names=["focal", "n1", "n2", "n3", "n4"],
        neutral_traits=[1, 2, 3, 4],
        calibrate=calibrate,
        cluster_ids=clusters,
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
    phylogeny_df, traits, clusters = _make_cherries(2_000)
    kwargs = dict(neutral_traits=[1, 2, 3, 4], calibrate=True)
    expected = alifestd_estimate_trait_gain_selection_asexual(
        phylogeny_df, trait_masks=traits, cluster_ids=clusters, **kwargs
    )

    order = np.random.default_rng(2).permutation(len(phylogeny_df))
    shuffled = phylogeny_df.iloc[order].reset_index(drop=True)
    shuffled["id"] += 7
    shuffled["ancestor_id"] += 7
    res = alifestd_estimate_trait_gain_selection_asexual(
        shuffled,
        trait_masks=traits[:, order],
        cluster_ids=clusters[order],
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
        trait_masks=np.array([[0, 0, 1, 1], [0, 0, 1, 0]], dtype=bool),
        min_leaf_origin_time=0.0,
        num_pendant_length_bins=4,
    )
    # both leaves carry trait 0 from above their parent, so both drop
    assert res["num_leaves"].tolist() == [0, 2]
    assert res["num_trait_present"].tolist() == [0, 1]
    assert np.isnan(res.loc[0, "selection"])
    assert np.isnan(res["gain_ratio"]).all()

    res = alifestd_estimate_trait_gain_selection_asexual(
        phylogeny_df,
        trait_masks=np.array([[0, 0, 1, 1], [0, 0, 1, 0]], dtype=bool),
        min_leaf_origin_time=3.0,
        num_pendant_length_bins=4,
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


def test_neutral_traits_normalized():
    phylogeny_df, traits, __ = _make_cherries(2_000)
    kwargs = dict(trait_masks=traits, calibrate=True)
    expected = alifestd_estimate_trait_gain_selection_asexual(
        phylogeny_df, neutral_traits=[1, 2, 3, 4], **kwargs
    )
    res = alifestd_estimate_trait_gain_selection_asexual(
        phylogeny_df, neutral_traits=[-1, -2, -3, -4, 4], **kwargs
    )
    pd.testing.assert_frame_equal(res, expected)

    with pytest.raises(ValueError):
        alifestd_estimate_trait_gain_selection_asexual(
            phylogeny_df, neutral_traits=[1, 1], **kwargs
        )


def test_unifurcations_bypassed():
    phylogeny_df, traits, __ = _make_cherries(2_000)
    expected = alifestd_estimate_trait_gain_selection_asexual(
        phylogeny_df, trait_masks=traits, neutral_traits=[1, 2, 3, 4]
    )

    # insert a knuckle midway along each pendant branch
    leaves = phylogeny_df.index[
        ~phylogeny_df["id"].isin(phylogeny_df["ancestor_id"])
    ]
    knuckles = phylogeny_df.loc[leaves].copy()
    knuckles["id"] = np.arange(len(knuckles)) + len(phylogeny_df)
    knuckles["origin_time"] = (
        knuckles["origin_time"]
        + phylogeny_df.loc[knuckles["ancestor_id"], "origin_time"].to_numpy()
    ) / 2
    knuckled_df = phylogeny_df.copy()
    knuckled_df.loc[leaves, "ancestor_id"] = knuckles["id"].to_numpy()
    knuckled_df = pd.concat([knuckled_df, knuckles], ignore_index=True)
    knuckled_traits = np.hstack([traits, traits[:, leaves]])

    res = alifestd_estimate_trait_gain_selection_asexual(
        knuckled_df, trait_masks=knuckled_traits, neutral_traits=[1, 2, 3, 4]
    )
    pd.testing.assert_frame_equal(res, expected)


def test_nan_origin_time_ignored():
    phylogeny_df, traits, __ = _make_cherries(2_000)
    kwargs = dict(neutral_traits=[1, 2, 3, 4], calibrate=True)
    expected = alifestd_estimate_trait_gain_selection_asexual(
        phylogeny_df, trait_masks=traits, **kwargs
    )

    # add a disconnected root with unknown origin time
    nan_df = pd.concat(
        [
            phylogeny_df,
            pd.DataFrame(
                {
                    "id": [len(phylogeny_df)],
                    "ancestor_id": [len(phylogeny_df)],
                    "origin_time": [np.nan],
                },
            ),
        ],
        ignore_index=True,
    )
    res = alifestd_estimate_trait_gain_selection_asexual(
        nan_df,
        trait_masks=np.hstack([traits, np.ones((5, 1), dtype=bool)]),
        **kwargs,
    )
    pd.testing.assert_frame_equal(res, expected)


def test_non_bool_trait_mask_raises():
    phylogeny_df, traits, __ = _make_cherries(100)
    with pytest.raises(NotImplementedError):
        alifestd_estimate_trait_gain_selection_asexual(
            phylogeny_df, trait_masks=np.where(traits, 1.0, np.nan)
        )


def test_trait_mask_length_mismatch_raises():
    phylogeny_df, traits, __ = _make_cherries(100)
    with pytest.raises(ValueError):
        alifestd_estimate_trait_gain_selection_asexual(
            phylogeny_df, trait_masks=traits[:, :-1]
        )


def test_null_cluster_ids_raise():
    phylogeny_df, traits, clusters = _make_cherries(100)
    clusters = clusters.astype(float)
    clusters[-1] = np.nan
    with pytest.raises(ValueError):
        alifestd_estimate_trait_gain_selection_asexual(
            phylogeny_df, trait_masks=traits, cluster_ids=clusters
        )


@pytest.mark.parametrize("parallel_backend", ["threading", "loky"])
def test_parallel_backend_matches_serial(parallel_backend: str):
    phylogeny_df, traits, clusters = _make_cherries(2_000)
    kwargs = dict(
        trait_masks=traits,
        neutral_traits=[1, 2, 3, 4],
        cluster_ids=clusters,
        num_bootstrap=4,
        seed=1,
    )
    expected = alifestd_estimate_trait_gain_selection_asexual(
        phylogeny_df, **kwargs
    )
    res = alifestd_estimate_trait_gain_selection_asexual(
        phylogeny_df, parallel_backend=parallel_backend, **kwargs
    )
    pd.testing.assert_frame_equal(res, expected)


def test_seed_reproducible():
    phylogeny_df, traits, clusters = _make_cherries(2_000)

    def run(seed):
        return alifestd_estimate_trait_gain_selection_asexual(
            phylogeny_df,
            trait_masks=traits,
            cluster_ids=clusters,
            num_bootstrap=4,
            seed=seed,
        )

    np.random.seed(5)
    expected_state = np.random.get_state()[1].copy()
    res1, res2, res3 = run(1), run(1), run(2)
    # seeded runs leave global rng state untouched
    np.testing.assert_array_equal(np.random.get_state()[1], expected_state)
    pd.testing.assert_frame_equal(res1, res2)
    assert not res1["selection_se"].equals(res3["selection_se"])


def test_bootstrap_without_neutral_traits():
    phylogeny_df, traits, clusters = _make_cherries(2_000)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no degenerate-replicate warning
        res = alifestd_estimate_trait_gain_selection_asexual(
            phylogeny_df,
            trait_masks=traits,
            cluster_ids=clusters,
            num_bootstrap=4,
            seed=1,
        )
    # gain ratio is undefined, but selection still gets standard errors
    assert np.isnan(res["gain_ratio"]).all()
    assert np.isnan(res["log_gain_ratio_se"]).all()
    assert np.isfinite(res["selection_se"]).all()
    assert (res["num_bootstrap_degenerate"] == 0).all()


@pytest.mark.parametrize(
    "param",
    [
        "num_pendant_length_bins",
        "num_calibration_pendant_length_groups",
        "num_calibration_time_groups",
    ],
)
def test_too_few_bins_raises(param: str):
    phylogeny_df, traits, __ = _make_cherries(100)
    with pytest.raises(ValueError):
        alifestd_estimate_trait_gain_selection_asexual(
            phylogeny_df, trait_masks=traits, **{param: 0}
        )


def test_single_bins():
    phylogeny_df, traits, __ = _make_cherries(2_000)
    res = alifestd_estimate_trait_gain_selection_asexual(
        phylogeny_df,
        trait_masks=traits,
        neutral_traits=[1, 2, 3, 4],
        calibrate=True,
        num_pendant_length_bins=1,
        num_calibration_pendant_length_groups=1,
        num_calibration_time_groups=1,
    )
    assert np.isfinite(res["gain_rate"]).all()


def test_bootstrap_rare_trait():
    phylogeny_df, traits, __ = _make_cherries(20_000)
    rare = np.zeros_like(traits[0])
    rare[np.flatnonzero(traits[1])[0]] = True
    with pytest.warns(UserWarning, match="bootstrap") as record:
        res = alifestd_estimate_trait_gain_selection_asexual(
            phylogeny_df,
            trait_masks=[rare, *traits[1:]],
            neutral_traits=[1, 2, 3, 4],
            num_bootstrap=50,
            seed=1,
        )
    # one cluster per leaf, so replicates often resample no rare carriers
    num_degenerate = res.loc[0, "num_bootstrap_degenerate"]
    assert 0 < num_degenerate < 50
    assert f"0 ({num_degenerate} of 50)" in str(record[0].message)
    assert (res.loc[1:, "num_bootstrap_degenerate"] == 0).all()
    assert np.isfinite(res["selection_se"]).all()
    assert np.isfinite(res["log_gain_ratio_se"]).all()


def test_tgs_bin_leaves():
    leaf_bins, group_ids = _tgs_bin_leaves(
        leaf_times=np.array([1.0, 2.0, 3.0, 4.0]),
        pendant_lengths=np.array([0.5, 1.0, 1.5, 2.0]),
        min_time=0.0,
        num_pendant_length_bins=4,
        num_calibration_pendant_length_groups=2,
        num_calibration_time_groups=2,
    )
    # time bins [0, 2), [2, 4]; length bins of width 0.5
    np.testing.assert_array_equal(leaf_bins, [1, 6, 7, 7])
    assert group_ids.shape == (2 * 4,)
    # leaves fall in length bins 1, 2, 3, 3; median 2.5 splits bins into
    # equal-occupancy groups {0, 1, 2} and {3}, crossed with time bins
    np.testing.assert_array_equal(group_ids, [0, 0, 0, 1, 2, 2, 2, 3])
