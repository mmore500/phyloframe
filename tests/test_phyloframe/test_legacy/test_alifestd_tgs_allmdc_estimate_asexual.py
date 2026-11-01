import numpy as np
import pandas as pd
import pytest

from phyloframe.legacy import (
    alifestd_tgs_allmdc_estimate_asexual,
    alifestd_tgs_pendantmdc_estimate_asexual,
)

SHIFT = 15.0


def _make_cherries(num_cherries: int = 40_000, seed: int = 1):
    """Root with cherries hanging off it. Each cherry's stem gains a trait
    under the clade model with a two-leaf shift of `SHIFT`, after which both
    leaves carry it; otherwise, each leaf gains it on its own pendant
    branch under the pendant model.

    Trait 0 has twice the neutral gain rate and a cost; traits 1-4 are
    neutral.
    """
    rng = np.random.default_rng(seed)
    num_nodes = 1 + 3 * num_cherries
    inner = np.arange(1, num_nodes, 3)
    ancestor_id = np.zeros(num_nodes, dtype=int)
    ancestor_id[inner + 1] = ancestor_id[inner + 2] = inner
    origin_time = np.zeros(num_nodes)
    stem = rng.uniform(0.5, 30, num_cherries)
    origin_time[inner] = stem
    pendant = rng.uniform(0.5, 60, (num_cherries, 2))
    origin_time[inner + 1] = stem + pendant[:, 0]
    origin_time[inner + 2] = stem + pendant[:, 1]

    traits = np.zeros((5, num_nodes), dtype=bool)
    for trait, (gain, selection) in enumerate(
        [(2e-3, -0.05)] + [(1e-3, 0.0)] * 4
    ):

        def p(length: np.ndarray) -> np.ndarray:
            if selection == 0:
                return gain * length
            return gain * np.expm1(selection * length) / selection

        stem_gain = rng.random(num_cherries) < p(stem) * np.exp(
            selection * SHIFT
        )
        own_gain = rng.random((num_cherries, 2)) < p(pendant)
        traits[trait, inner + 1] = stem_gain | own_gain[:, 0]
        traits[trait, inner + 2] = stem_gain | own_gain[:, 1]

    phylogeny_df = pd.DataFrame(
        {
            "id": range(num_nodes),
            "ancestor_id": ancestor_id,
            "origin_time": origin_time,
        },
    )
    return phylogeny_df, traits


@pytest.mark.parametrize("calibrate", [False, True])
def test_recovers_gain_selection_shift(calibrate: bool):
    phylogeny_df, traits = _make_cherries()
    original_df = phylogeny_df.copy()
    res = alifestd_tgs_allmdc_estimate_asexual(
        phylogeny_df,
        trait_masks=traits,
        trait_names=["focal", "n1", "n2", "n3", "n4"],
        neutral_traits=[1, 2, 3, 4],
        calibrate=calibrate,
        num_bootstrap=10,
        seed=1,
    )
    assert original_df.equals(phylogeny_df)
    assert res["trait"].tolist() == ["focal", "n1", "n2", "n3", "n4"]
    assert (res["num_clades"] == 40_000).all()
    assert (res["num_clade_gains"] > 0).all()

    focal = res.iloc[0]
    assert focal["gain_ratio"] == pytest.approx(2.0, rel=0.25)
    assert focal["selection"] == pytest.approx(-0.05, abs=0.02)
    assert focal["gain_ratio_p"] < 1e-3
    assert focal["selection_p"] < 1e-3
    # independent gains on both leaves of a cherry also look like a stem
    # gain under parsimony, so the shift comes out somewhat short
    assert 8.0 < focal["clade_size_shift_2"] < 20.0
    assert np.isfinite(res["clade_size_shift_2_se"]).all()
    assert res["clade_size_shift_2"].nunique() == 1
    # cherries only, so there are no larger clade-size groups
    assert [c for c in res.columns if c.startswith("clade_size_shift")] == [
        "clade_size_shift_2",
        "clade_size_shift_2_se",
    ]

    neutral = res.iloc[1:]
    assert neutral["gain_ratio"].tolist() == pytest.approx([1.0] * 4, rel=0.3)
    assert (neutral["selection"].abs() < 0.02).all()


@pytest.mark.parametrize("calibrate", [False, True])
def test_matches_pendant_without_inner_branches(calibrate: bool):
    rng = np.random.default_rng(3)
    num_leaves = 20_000
    origin_time = np.concatenate([[0.0], rng.uniform(0.5, 60, num_leaves)])
    phylogeny_df = pd.DataFrame(
        {
            "id": range(num_leaves + 1),
            "ancestor_id": 0,
            "origin_time": origin_time,
        },
    )
    length = origin_time[1:]
    traits = np.array(
        [
            np.r_[False, rng.random(num_leaves) < 2e-3 * length]
            for __ in range(5)
        ],
    )
    kwargs = dict(
        trait_masks=traits, neutral_traits=[1, 2, 3, 4], calibrate=calibrate
    )
    res = alifestd_tgs_allmdc_estimate_asexual(phylogeny_df, **kwargs)
    expected = alifestd_tgs_pendantmdc_estimate_asexual(phylogeny_df, **kwargs)
    for column in ("num_leaves", "num_trait_present"):
        assert res[column].tolist() == expected[column].tolist()
    for column in ("gain_rate", "selection", "gain_ratio"):
        assert res[column].to_numpy() == pytest.approx(
            expected[column].to_numpy(), rel=1e-4, abs=1e-8
        )
    assert (res["num_clades"] == 0).all()
    assert not any(c.startswith("clade_size_shift") for c in res.columns)


def test_counts():
    #       0
    #     1   4
    #    2 3
    phylogeny_df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3, 4],
            "ancestor_id": [0, 0, 1, 1, 0],
            "origin_time": [0.0, 1.0, 3.0, 4.0, 2.0],
        },
    )
    res = alifestd_tgs_allmdc_estimate_asexual(
        phylogeny_df,
        trait_masks=np.array(
            [[0, 0, 1, 1, 0], [0, 0, 1, 0, 0], [0, 0, 1, 1, 1]],
            dtype=bool,
        ),
    )
    # trait 0: gained on branch 1, whose clade carries it at both leaves,
    # so leaves 2 and 3 started with it; leaf 4 started without it
    assert res.loc[0, ["num_leaves", "num_trait_present"]].tolist() == [1, 0]
    assert res.loc[0, ["num_clades", "num_clade_gains"]].tolist() == [1, 1]
    # trait 1: gained on leaf 2's pendant branch only
    assert res.loc[1, ["num_leaves", "num_trait_present"]].tolist() == [3, 1]
    assert res.loc[1, ["num_clades", "num_clade_gains"]].tolist() == [1, 0]
    # trait 2: every leaf carries it, so it predates every branch
    assert res.loc[2, ["num_leaves", "num_trait_present"]].tolist() == [0, 0]
    assert res.loc[2, ["num_clades", "num_clade_gains"]].tolist() == [0, 0]
    assert np.isnan(res.loc[2, "selection"])


def test_row_order_invariant():
    phylogeny_df, traits = _make_cherries(2_000)
    kwargs = dict(neutral_traits=[1, 2, 3, 4], calibrate=True)
    expected = alifestd_tgs_allmdc_estimate_asexual(
        phylogeny_df, trait_masks=traits, **kwargs
    )

    order = np.random.default_rng(2).permutation(len(phylogeny_df))
    shuffled = phylogeny_df.iloc[order].reset_index(drop=True)
    shuffled["id"] += 7
    shuffled["ancestor_id"] += 7
    res = alifestd_tgs_allmdc_estimate_asexual(
        shuffled, trait_masks=traits[:, order], **kwargs
    )
    pd.testing.assert_frame_equal(res, expected, rtol=1e-4)


def test_clade_size_groups_per_doubling_validated():
    phylogeny_df, traits = _make_cherries(100)
    with pytest.raises(ValueError):
        alifestd_tgs_allmdc_estimate_asexual(
            phylogeny_df, trait_masks=traits, clade_size_groups_per_doubling=0
        )


@pytest.mark.parametrize(
    "groups_per_doubling, expected",
    [
        (1, ["clade_size_shift_2_3", "clade_size_shift_4_7"]),
        (
            2,
            [
                "clade_size_shift_2",
                "clade_size_shift_3",
                "clade_size_shift_4_5",
                "clade_size_shift_6_7",
            ],
        ),
    ],
)
def test_clade_size_groups_per_doubling(groups_per_doubling: int, expected):
    # root over a 7-leaf caterpillar, so clades of 2 to 7 leaves
    ancestor_id = [0, 0, 1, 1, 3, 3, 5, 5, 7, 7, 9, 9, 11, 11]
    phylogeny_df = pd.DataFrame(
        {
            "id": range(14),
            "ancestor_id": ancestor_id,
            "origin_time": np.arange(14, dtype=float),
        },
    )
    res = alifestd_tgs_allmdc_estimate_asexual(
        phylogeny_df,
        trait_masks=[np.zeros(14, dtype=bool)],
        clade_size_groups_per_doubling=groups_per_doubling,
    )
    assert [
        c for c in res.columns if c.startswith("clade_size_shift")
    ] == expected


def test_calibrate_requires_neutral():
    phylogeny_df, traits = _make_cherries(100)
    with pytest.raises(ValueError):
        alifestd_tgs_allmdc_estimate_asexual(
            phylogeny_df,
            trait_masks=traits,
            calibrate=True,
            neutral_traits=[1],
        )


def test_trait_mask_validated():
    phylogeny_df, traits = _make_cherries(100)
    with pytest.raises(NotImplementedError):
        alifestd_tgs_allmdc_estimate_asexual(
            phylogeny_df, trait_masks=traits.astype(int)
        )
    with pytest.raises(ValueError):
        alifestd_tgs_allmdc_estimate_asexual(
            phylogeny_df, trait_masks=traits[:, :-1]
        )
    with pytest.raises(ValueError):
        alifestd_tgs_allmdc_estimate_asexual(
            phylogeny_df, trait_masks=traits, trait_names=["a"]
        )
