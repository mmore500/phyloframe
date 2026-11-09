import typing

import numpy as np
import pandas as pd
import polars as pl
import polars.testing as pltest
import pytest

from phyloframe.legacy import (
    alifestd_mark_leaves_polars,
    alifestd_tgs_estimate_polars,
    alifestd_to_working_format,
)

from ._impl import add_trait_transitions, mark_tgs_calibration_columns
from .test_alifestd_tgs_estimate_polars_leaf_mask import _estimate_leaves


def _estimate(
    estimator: typing.Callable,
    phylogeny_df: pd.DataFrame,
    calibrate: bool = False,
    calibrate_size_ratio: float = 2.0,
    **kwargs,
) -> pd.DataFrame:
    """Run a polars estimator on a pandas phylogeny, returning pandas. If
    `calibrate`, calibrates by partitions and bins from
    `mark_tgs_calibration_columns`."""
    df_pl = pl.from_pandas(alifestd_to_working_format(phylogeny_df))
    if calibrate:
        df_pl = mark_tgs_calibration_columns(
            df_pl.drop("ancestor_list", strict=False),
            neutral_traits=kwargs["neutral_traits"],
            size_ratio=calibrate_size_ratio,
        )
        kwargs["calibration_partition_by"] = "clade_size_class"
        kwargs["calibration_compare_by_bins"] = "joint_quantile_bin"
    return estimator(df_pl, **kwargs).to_pandas()


def _estimate_all(
    phylogeny_df: pd.DataFrame,
    mutate: bool = False,
    estimator: typing.Callable = alifestd_tgs_estimate_polars,
    **kwargs,
) -> pd.DataFrame:
    return _estimate(estimator, phylogeny_df, **kwargs)


def _estimate_leaves_only(
    phylogeny_df: pd.DataFrame, mutate: bool = False, **kwargs
) -> pd.DataFrame:
    return _estimate(_estimate_leaves, phylogeny_df, **kwargs)


def _estimate_from_masks(
    phylogeny_df: pd.DataFrame,
    mutate: bool = False,
    *,
    trait_masks: typing.Sequence[np.ndarray],
    trait_names: typing.Optional[typing.Sequence[str]] = None,
    neutral_traits: typing.Sequence[int] = (1, 2, 3, 4),
    **kwargs,
) -> pd.DataFrame:
    """Run the estimator on transitions reconstructed from trait masks,
    named `trait_names`, with `neutral_traits` given by index."""
    phylogeny_df, names = add_trait_transitions(
        phylogeny_df, trait_masks, trait_names
    )
    kwargs["neutral_traits"] = [names[i] for i in neutral_traits]
    return _estimate_all(
        phylogeny_df, mutate, trait_transitions=names, **kwargs
    )


def _make_cherries(num_cherries: int = 40_000, seed: int = 1):
    """Root with cherries hanging off it. Each cherry's stem gains a trait
    under the full model, surviving until its later leaf is sampled,
    after which both leaves carry it; otherwise, each leaf gains it on its
    own pendant branch under the pendant model.

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
    for trait, (gain, selection_coef) in enumerate(
        [(2e-3, -0.05)] + [(1e-3, 0.0)] * 4
    ):

        def p(length: np.ndarray) -> np.ndarray:
            if selection_coef == 0:
                return gain * length
            return gain * np.expm1(selection_coef * length) / selection_coef

        survival = pendant.max(axis=1)
        stem_gain = rng.random(num_cherries) < p(stem) * np.exp(
            selection_coef * survival
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
def test_recovers_gain_selection(calibrate: bool):
    phylogeny_df, traits = _make_cherries()
    res = _estimate_from_masks(
        phylogeny_df,
        trait_masks=traits,
        trait_names=["focal", "n1", "n2", "n3", "n4"],
        neutral_traits=[1, 2, 3, 4],
        calibrate=calibrate,
    )
    assert res["trait"].tolist() == ["focal", "n1", "n2", "n3", "n4"]
    assert (res["num_inner_episodes"] == 40_000).all()
    assert (res["num_inner_gains"] > 0).all()

    focal = res.iloc[0]
    assert focal["gain_ratio"] == pytest.approx(2.0, rel=0.25)
    assert focal["selection_coef"] == pytest.approx(-0.05, abs=0.02)
    assert not any("shift" in c for c in res.columns)

    neutral = res.iloc[1:]
    assert neutral["gain_ratio"].tolist() == pytest.approx([1.0] * 4, rel=0.3)
    assert (neutral["selection_coef"].abs() < 0.02).all()


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
    phylogeny_df, names = add_trait_transitions(phylogeny_df, traits)
    kwargs = dict(
        trait_transitions=names,
        neutral_traits=names[1:],
        calibrate=calibrate,
    )
    res = _estimate_all(phylogeny_df, **kwargs)
    expected = _estimate_leaves_only(phylogeny_df, **kwargs)
    for column in ("num_leaf_episodes", "num_leaf_gains"):
        assert res[column].tolist() == expected[column].tolist()
    for column in ("gain_rate", "selection_coef", "gain_ratio"):
        assert res[column].to_numpy() == pytest.approx(
            expected[column].to_numpy(), rel=1e-4, abs=1e-8
        )
    assert (res["num_inner_episodes"] == 0).all()


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
    res = _estimate_from_masks(
        phylogeny_df,
        trait_masks=np.array(
            [[0, 0, 1, 1, 0], [0, 0, 1, 0, 0]] + [[0, 0, 0, 0, 0]] * 2,
            dtype=bool,
        ),
        neutral_traits=[2, 3],
    )
    # trait 0: gained on branch 1, whose clade carries it at both leaves,
    # so leaves 2 and 3 started with it; leaf 4 started without it
    assert res.loc[0, ["num_leaf_episodes", "num_leaf_gains"]].tolist() == [
        1,
        0,
    ]
    assert res.loc[0, ["num_inner_episodes", "num_inner_gains"]].tolist() == [
        1,
        1,
    ]
    # trait 1: gained on leaf 2's pendant branch only
    assert res.loc[1, ["num_leaf_episodes", "num_leaf_gains"]].tolist() == [
        3,
        1,
    ]
    assert res.loc[1, ["num_inner_episodes", "num_inner_gains"]].tolist() == [
        1,
        0,
    ]


def test_root_present_trait_raises():
    phylogeny_df, traits = _make_cherries(100)
    # every leaf carries trait 0, so the root does too
    traits[0] = traits[0] | ~np.isin(
        phylogeny_df["id"], phylogeny_df["ancestor_id"]
    )
    with pytest.raises(NotImplementedError, match="present at a root"):
        _estimate_from_masks(phylogeny_df, trait_masks=traits)


def test_row_order_invariant():
    phylogeny_df, traits = _make_cherries(2_000)
    kwargs = dict(neutral_traits=[1, 2, 3, 4], calibrate=True)
    expected = _estimate_from_masks(phylogeny_df, trait_masks=traits, **kwargs)

    order = np.random.default_rng(2).permutation(len(phylogeny_df))
    shuffled = phylogeny_df.iloc[order].reset_index(drop=True)
    shuffled["id"] += 7
    shuffled["ancestor_id"] += 7
    res = _estimate_from_masks(
        shuffled, trait_masks=traits[:, order], **kwargs
    )
    pd.testing.assert_frame_equal(res, expected, rtol=1e-4)


@pytest.mark.parametrize("ratio", [1.5, 2.0, 1e9])
def test_calibrate_size_ratio(ratio: float):
    phylogeny_df, traits = _make_cherries(10_000)
    res = _estimate_from_masks(
        phylogeny_df,
        trait_masks=traits,
        calibrate=True,
        calibrate_size_ratio=ratio,
    )
    assert res.loc[0, "selection_coef"] < 0
    assert np.isfinite(res["gain_ratio"]).all()


def test_calibration_compare_by_vals_clade_size():
    phylogeny_df, traits = _make_cherries(10_000)
    kwargs = dict(trait_masks=traits, calibrate=True)
    default = _estimate_from_masks(phylogeny_df, **kwargs)
    res = _estimate_from_masks(
        phylogeny_df,
        calibration_compare_by_vals=["branch_length", "clade_size"],
        **kwargs,
    )
    assert not res.equals(default)
    assert np.isfinite(res["gain_ratio"]).all()


@pytest.mark.parametrize("neutral_traits", [[], [1], [1, 1]])
def test_too_few_neutral_traits_raise(neutral_traits):
    phylogeny_df, traits = _make_cherries(100)
    with pytest.raises(NotImplementedError, match="at least two"):
        _estimate_from_masks(
            phylogeny_df, trait_masks=traits, neutral_traits=neutral_traits
        )


def test_neutral_trait_index_raises():
    phylogeny_df, traits = _make_cherries(100)
    phylogeny_df, names = add_trait_transitions(phylogeny_df, traits)
    with pytest.raises(TypeError):
        _estimate_all(
            phylogeny_df, trait_transitions=names, neutral_traits=[1]
        )


def test_input_not_mutated():
    phylogeny_df, traits = _make_cherries(100)
    phylogeny_df, names = add_trait_transitions(phylogeny_df, traits)
    original_df = phylogeny_df.copy()
    _estimate_all(
        phylogeny_df, trait_transitions=names, neutral_traits=names[1:]
    )
    pd.testing.assert_frame_equal(phylogeny_df, original_df)


def _make_cherries_with_transitions(num_cherries: int = 100):
    phylogeny_df, traits = _make_cherries(num_cherries)
    return add_trait_transitions(phylogeny_df, traits)


def test_missing_transitions_column_raises():
    phylogeny_df, names = _make_cherries_with_transitions()
    with pytest.raises(pl.exceptions.ColumnNotFoundError, match="absent"):
        _estimate_all(
            phylogeny_df,
            trait_transitions=["absent", *names[1:]],
            neutral_traits=names[1:],
        )


def test_missing_calibration_column_raises():
    phylogeny_df, names = _make_cherries_with_transitions()
    with pytest.raises(pl.exceptions.ColumnNotFoundError, match="absent"):
        _estimate_all(
            phylogeny_df,
            trait_transitions=names,
            neutral_traits=names[1:],
            calibration_partition_by="absent",
        )


def test_float_transitions_raise():
    phylogeny_df, names = _make_cherries_with_transitions()
    phylogeny_df[names[0]] = phylogeny_df[names[0]].astype(float)
    with pytest.raises(NotImplementedError):
        _estimate_all(
            phylogeny_df, trait_transitions=names, neutral_traits=names[1:]
        )


@pytest.mark.parametrize("value", [2, -2])
def test_invalid_transitions_raise(value: int):
    phylogeny_df, names = _make_cherries_with_transitions()
    phylogeny_df.loc[5, names[0]] = value
    with pytest.raises(ValueError):
        _estimate_all(
            phylogeny_df, trait_transitions=names, neutral_traits=names[1:]
        )


def test_repeated_gain_raises():
    phylogeny_df, names = _make_cherries_with_transitions()
    # gains on cherry 1's stem and on one of its leaves give state 2 there
    phylogeny_df["trait"] = 0
    phylogeny_df.loc[[1, 2], "trait"] = 1
    with pytest.raises(ValueError):
        _estimate_all(
            phylogeny_df,
            trait_transitions=["trait", *names[1:]],
            neutral_traits=names[1:],
        )


def test_losses_not_supported():
    phylogeny_df, names = _make_cherries_with_transitions()
    phylogeny_df["trait"] = 0
    phylogeny_df.loc[1, "trait"] = 1
    phylogeny_df.loc[2, "trait"] = -1
    with pytest.raises(NotImplementedError, match="loss"):
        _estimate_all(
            phylogeny_df,
            trait_transitions=["trait", *names[1:]],
            neutral_traits=names[1:],
        )


def test_trait_named_by_columns():
    phylogeny_df, names = _make_cherries_with_transitions()
    phylogeny_df = phylogeny_df.rename(columns=dict(zip(names, "abcde")))
    res = _estimate_all(
        phylogeny_df,
        trait_transitions=list("abcde"),
        neutral_traits=["b", "c"],
    )
    assert res["trait"].tolist() == list("abcde")


apply_params = pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)


def _make_cherries_polars(num_cherries: int) -> pl.DataFrame:
    phylogeny_df, traits = _make_cherries(num_cherries)
    phylogeny_df, names = add_trait_transitions(phylogeny_df, traits)
    return pl.from_pandas(
        phylogeny_df.rename(
            columns=dict(zip(names, ["focal", "n1", "n2", "n3", "n4"]))
        ).drop(columns=["ancestor_list"], errors="ignore"),
    )


@apply_params
def test_empty(apply: typing.Callable):
    df_pl = apply(
        pl.DataFrame(
            {
                "id": pl.Series([], dtype=pl.Int64),
                "ancestor_id": pl.Series([], dtype=pl.Int64),
                "origin_time": pl.Series([], dtype=pl.Float64),
                **{t: pl.Series([], dtype=pl.Int8) for t in ("a", "b", "c")},
            }
        ),
    )
    with pytest.raises(ValueError, match="no exposure episodes"):
        alifestd_tgs_estimate_polars(
            df_pl,
            trait_transitions=["a", "b", "c"],
            neutral_traits=["b", "c"],
        )


def test_expression_transitions_raise():
    df_pl = _make_cherries_polars(2_000)
    with pytest.raises(TypeError):
        alifestd_tgs_estimate_polars(
            df_pl, trait_transitions=[pl.col("focal")]
        )


def test_not_asexual():
    df_pl = pl.DataFrame(
        {
            "id": [0, 1],
            "ancestor_list": ["[none]", "[0]"],
            "origin_time": [0.0, 1.0],
            "a": [0, 1],
            "b": [0, 0],
            "c": [0, 0],
        },
    ).with_columns(ancestor_list=pl.lit("[0, 1]"))
    with pytest.raises(NotImplementedError, match="asexual"):
        alifestd_tgs_estimate_polars(
            df_pl, trait_transitions=["a", "b", "c"], neutral_traits=["b", "c"]
        )


_MASK_KWARGS = dict(
    trait_transitions=["focal", "n1", "n2", "n3", "n4"],
    neutral_traits=["n1", "n2", "n3", "n4"],
)


def test_episode_mask_all_true_matches_unmasked():
    df_pl = _make_cherries_polars(2_000).with_columns(keep=pl.lit(True))
    pltest.assert_frame_equal(
        alifestd_tgs_estimate_polars(
            df_pl, episode_mask="keep", **_MASK_KWARGS
        ),
        alifestd_tgs_estimate_polars(df_pl, **_MASK_KWARGS),
    )


def test_single_string_calibration_compare_by_vals():
    df_pl = _make_cherries_polars(2_000)
    pltest.assert_frame_equal(
        alifestd_tgs_estimate_polars(
            df_pl, calibration_compare_by_vals="origin_time", **_MASK_KWARGS
        ),
        alifestd_tgs_estimate_polars(
            df_pl, calibration_compare_by_vals=["origin_time"], **_MASK_KWARGS
        ),
    )


def test_episode_mask_leaves_counts_no_inner_episodes():
    df_pl = alifestd_mark_leaves_polars(_make_cherries_polars(2_000))
    res = alifestd_tgs_estimate_polars(
        df_pl, episode_mask="is_leaf", **_MASK_KWARGS
    )
    assert (res["num_inner_episodes"] == 0).all()
    assert (res["num_leaf_episodes"] > 0).all()


@pytest.mark.parametrize(
    "mask", [pl.lit(1), pl.lit(None, dtype=pl.Boolean)], ids=["int", "null"]
)
def test_invalid_episode_mask_raises(mask: pl.Expr):
    df_pl = _make_cherries_polars(500).with_columns(mask.alias("keep"))
    with pytest.raises(NotImplementedError, match="episode mask"):
        alifestd_tgs_estimate_polars(
            df_pl, episode_mask="keep", **_MASK_KWARGS
        )


def test_missing_episode_mask_raises():
    with pytest.raises(pl.exceptions.ColumnNotFoundError):
        alifestd_tgs_estimate_polars(
            _make_cherries_polars(500), episode_mask="keep", **_MASK_KWARGS
        )


def test_episode_mask_excluding_all_raises():
    df_pl = _make_cherries_polars(500).with_columns(keep=pl.lit(False))
    with pytest.raises(ValueError, match="no exposure episodes"):
        alifestd_tgs_estimate_polars(
            df_pl, episode_mask="keep", **_MASK_KWARGS
        )
