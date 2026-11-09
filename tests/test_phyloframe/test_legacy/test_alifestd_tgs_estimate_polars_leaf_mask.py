import typing

import numpy as np
import pandas as pd
import polars as pl
import polars.testing as pltest
import pytest

from phyloframe.legacy import (
    alifestd_assign_contiguous_ids_polars,
    alifestd_collapse_unifurcations_polars,
    alifestd_mark_leaves_polars,
    alifestd_mark_lineage_cumsum_polars,
    alifestd_mark_lineage_delta_polars,
    alifestd_tgs_estimate_polars,
    alifestd_to_working_format,
    alifestd_try_add_ancestor_id_col_polars,
)
from phyloframe.legacy._impl._tgs_select_partitioned_calibration_neighborhoods import (
    _tgs_select_calibration_neighborhoods,
)

from ._impl import add_trait_transitions, mark_tgs_calibration_columns


def _with_leaf_mask(phylogeny_df):
    """Mark leaves, `is_leaf`, to mask exposure episodes to leaf pendant
    branches; unchanged if not asexual, so the estimator raises."""
    phylogeny_df = alifestd_try_add_ancestor_id_col_polars(phylogeny_df)
    if "ancestor_id" not in phylogeny_df.lazy().collect_schema().names():
        return phylogeny_df
    return alifestd_mark_leaves_polars(phylogeny_df)


def _estimate_leaves(phylogeny_df, **kwargs):
    """Estimate from leaf pendant branches only, via `episode_mask`."""
    return alifestd_tgs_estimate_polars(
        _with_leaf_mask(phylogeny_df), episode_mask="is_leaf", **kwargs
    )


def _estimate_pandas(
    phylogeny_df: pd.DataFrame,
    mutate: bool = False,
    calibrate: bool = False,
    calibrate_max_num_bins: int = 4096,
    estimator: typing.Callable = _estimate_leaves,
    **kwargs,
) -> pd.DataFrame:
    """Run the polars estimator on a pandas phylogeny, returning pandas. If
    `calibrate`, calibrates by bins from `mark_tgs_calibration_columns`."""
    df_pl = pl.from_pandas(alifestd_to_working_format(phylogeny_df))
    if calibrate:
        df_pl = mark_tgs_calibration_columns(
            df_pl.drop("ancestor_list", strict=False),
            neutral_traits=kwargs["neutral_traits"],
            max_num_bins=calibrate_max_num_bins,
        )
        kwargs["calibration_compare_by_bins"] = "joint_quantile_bin"
    return estimator(df_pl, **kwargs).to_pandas()


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
    return _estimate_pandas(
        phylogeny_df, mutate, trait_transitions=names, **kwargs
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
    for trait, (gain, selection_coef) in enumerate(
        [(2e-3, -0.05)] + [(1e-3, 0.0)] * 4
    ):
        p = (
            gain * length
            if selection_coef == 0
            else gain * np.expm1(selection_coef * length) / selection_coef
        )
        traits[trait, leaves] = rng.random(len(leaves)) < p

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
    assert (res["num_leaf_episodes"] <= 80_000).all()

    focal = res.iloc[0]
    assert focal["gain_ratio"] == pytest.approx(2.0, rel=0.2)
    assert focal["selection_coef"] == pytest.approx(-0.05, abs=0.02)

    neutral = res.iloc[1:]
    # gain rate trades off against the selection coefficient, so with ~2k
    # carriers per trait, neutral gain ratios scatter by ~20%
    assert neutral["gain_ratio"].tolist() == pytest.approx([1.0] * 4, rel=0.3)
    assert (neutral["selection_coef"].abs() < 0.02).all()


def test_row_order_invariant():
    phylogeny_df, traits = _make_cherries(2_000)
    kwargs = dict(neutral_traits=[1, 2, 3, 4], calibrate=True)
    expected = _estimate_from_masks(phylogeny_df, trait_masks=traits, **kwargs)

    order = np.random.default_rng(2).permutation(len(phylogeny_df))
    shuffled = phylogeny_df.iloc[order].reset_index(drop=True)
    shuffled["id"] += 7
    shuffled["ancestor_id"] += 7
    res = _estimate_from_masks(
        shuffled,
        trait_masks=traits[:, order],
        **kwargs,
    )
    pd.testing.assert_frame_equal(res, expected, rtol=1e-6)


def test_drops_inherited_and_rootward():
    #       0
    #     1   4
    #   2   3
    phylogeny_df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3, 4],
            "ancestor_id": [0, 0, 1, 1, 0],
            "origin_time": [0.0, 1.0, 2.0, 4.0, 3.0],
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
    # leaves 2 and 3 carry trait 0 from above their parent, so both drop;
    # leaf 4 starts without it
    assert res["num_leaf_episodes"].tolist() == [1, 3, 3, 3]
    assert res["num_leaf_gains"].tolist() == [0, 1, 0, 0]
    assert np.isnan(res.loc[0, "selection_coef"])
    # the neutral traits never gain, so gain ratios are undefined
    assert np.isnan(res["gain_ratio"]).all()


@pytest.mark.parametrize("neutral_traits", [[], [1], [1, 1]])
def test_too_few_neutral_traits_raise(neutral_traits):
    phylogeny_df, traits = _make_cherries(100)
    with pytest.raises(NotImplementedError, match="at least two"):
        _estimate_from_masks(
            phylogeny_df, trait_masks=traits, neutral_traits=neutral_traits
        )


def test_neutral_traits_normalized():
    phylogeny_df, traits = _make_cherries(2_000)
    kwargs = dict(trait_masks=traits, calibrate=True)
    expected = _estimate_from_masks(
        phylogeny_df, neutral_traits=[1, 2, 3, 4], **kwargs
    )
    res = _estimate_from_masks(
        phylogeny_df, neutral_traits=[4, 3, 2, 1, 4], **kwargs
    )
    pd.testing.assert_frame_equal(res, expected)


def test_unifurcations_raise():
    phylogeny_df, traits = _make_cherries(2_000)
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
    with pytest.raises(NotImplementedError, match="unifurcations"):
        _estimate_from_masks(
            knuckled_df,
            trait_masks=knuckled_traits,
            neutral_traits=[1, 2, 3, 4],
        )


def test_nan_origin_time_ignored():
    phylogeny_df, traits = _make_cherries(2_000)
    # no calibration bins, as alifestd_mark_joint_quantile_bins_polars
    # rejects nan values
    kwargs = dict(neutral_traits=[1, 2, 3, 4])
    expected = _estimate_from_masks(phylogeny_df, trait_masks=traits, **kwargs)

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
    res = _estimate_from_masks(
        nan_df,
        trait_masks=np.hstack([traits, np.zeros((5, 1), dtype=bool)]),
        **kwargs,
    )
    pd.testing.assert_frame_equal(res, expected)


def test_input_not_mutated():
    phylogeny_df, traits = _make_cherries(100)
    phylogeny_df, names = add_trait_transitions(phylogeny_df, traits)
    original_df = phylogeny_df.copy()
    _estimate_pandas(
        phylogeny_df, trait_transitions=names, neutral_traits=names[1:]
    )
    pd.testing.assert_frame_equal(phylogeny_df, original_df)


def _make_cherries_with_transitions(num_cherries: int = 100):
    phylogeny_df, traits = _make_cherries(num_cherries)
    return add_trait_transitions(phylogeny_df, traits)


def test_missing_transitions_column_raises():
    phylogeny_df, names = _make_cherries_with_transitions()
    with pytest.raises(pl.exceptions.ColumnNotFoundError, match="absent"):
        _estimate_pandas(
            phylogeny_df,
            trait_transitions=["absent", *names[1:]],
            neutral_traits=names[1:],
        )


def test_float_transitions_raise():
    phylogeny_df, names = _make_cherries_with_transitions()
    phylogeny_df[names[0]] = phylogeny_df[names[0]].astype(float)
    with pytest.raises(NotImplementedError):
        _estimate_pandas(
            phylogeny_df, trait_transitions=names, neutral_traits=names[1:]
        )


@pytest.mark.parametrize("value", [2, -2])
def test_invalid_transitions_raise(value: int):
    phylogeny_df, names = _make_cherries_with_transitions()
    phylogeny_df.loc[5, names[0]] = value
    with pytest.raises(ValueError):
        _estimate_pandas(
            phylogeny_df, trait_transitions=names, neutral_traits=names[1:]
        )


def test_repeated_gain_raises():
    phylogeny_df, names = _make_cherries_with_transitions()
    # gains on cherry 1's stem and on one of its leaves give state 2 there
    phylogeny_df["trait"] = 0
    phylogeny_df.loc[[1, 2], "trait"] = 1
    with pytest.raises(ValueError):
        _estimate_pandas(
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
        _estimate_pandas(
            phylogeny_df,
            trait_transitions=["trait", *names[1:]],
            neutral_traits=names[1:],
        )


def test_trait_named_by_columns():
    phylogeny_df, names = _make_cherries_with_transitions()
    phylogeny_df = phylogeny_df.rename(columns=dict(zip(names, "abcde")))
    res = _estimate_pandas(
        phylogeny_df,
        trait_transitions=list("abcde"),
        neutral_traits=["b", "c"],
    )
    assert res["trait"].tolist() == list("abcde")


def test_tgs_select_calibration_neighborhoods():
    rng = np.random.default_rng(1)
    bin_coords = rng.random((256, 2))
    counts = pl.DataFrame(
        {
            "trait_id": np.repeat(np.arange(3), 256),
            "bin_id": np.tile(np.arange(256), 3),
            "num_trials": 100.0,
            "num_successes": rng.binomial(100, 0.01, 3 * 256).astype(float),
            "length_sum": 1000.0,
        },
    )
    res = _tgs_select_calibration_neighborhoods(
        bin_coords=bin_coords,
        neighborhood_fractions=(0.01, 0.1, 0.5),
        trait_bin_counts_df=counts,
        neutral_trait_ids=[0, 1, 2],
        trait_names=["a", "b", "c"],
        max_rate_ratio=np.inf,
    )
    assert res.shape == (256, 256)
    # each cell belongs to its own neighborhood
    assert (res.diagonal() == 1).all()
    # neighborhoods are the same size
    assert len(set(np.diff(res.indptr))) == 1


@pytest.mark.parametrize("fractions", [(), (0.0, 0.1), (0.1, 1.5)])
def test_invalid_neighborhood_fractions_raise(fractions):
    phylogeny_df, traits = _make_cherries(100)
    with pytest.raises(ValueError):
        _estimate_from_masks(
            phylogeny_df,
            trait_masks=traits,
            calibrate_neighborhood_fractions=fractions,
        )


def test_neighborhood_fractions_single_candidate():
    phylogeny_df, traits = _make_cherries(2_000)
    res = _estimate_from_masks(
        phylogeny_df,
        trait_masks=traits,
        neutral_traits=[1, 2, 3, 4],
        calibrate=True,
        calibrate_neighborhood_fractions=[0.1],
    )
    assert np.isfinite(res["selection_coef"]).all()


def test_neutral_trait_not_among_transitions_raises():
    phylogeny_df, traits = _make_cherries(100)
    phylogeny_df, names = add_trait_transitions(phylogeny_df, traits)
    with pytest.raises(ValueError):
        _estimate_pandas(
            phylogeny_df, trait_transitions=names, neutral_traits=["absent"]
        )


def test_neutral_trait_index_raises():
    phylogeny_df, traits = _make_cherries(100)
    phylogeny_df, names = add_trait_transitions(phylogeny_df, traits)
    with pytest.raises(TypeError):
        _estimate_pandas(
            phylogeny_df, trait_transitions=names, neutral_traits=[1]
        )


@pytest.mark.parametrize("calibrate_max_num_bins", [1, 64, 2**20])
def test_calibrate_max_num_bins(calibrate_max_num_bins: int):
    # more bins than leaves allow are capped, keeping 4 leaves each
    phylogeny_df, traits = _make_cherries(2_000)
    res = _estimate_from_masks(
        phylogeny_df,
        trait_masks=traits,
        calibrate=True,
        calibrate_max_num_bins=calibrate_max_num_bins,
    )
    assert np.isfinite(res["selection_coef"]).all()


def test_calibration_compare_by_vals():
    phylogeny_df, traits = _make_cherries(2_000)
    kwargs = dict(trait_masks=traits, calibrate=True)
    default = _estimate_from_masks(phylogeny_df, **kwargs)
    explicit = _estimate_from_masks(
        phylogeny_df,
        calibration_compare_by_vals=["branch_length", "origin_time"],
        **kwargs,
    )
    pd.testing.assert_frame_equal(explicit, default)
    # a phylogeny column, here a copy of origin time, compares like one
    by_column = _estimate_from_masks(
        phylogeny_df.assign(t=phylogeny_df["origin_time"]),
        calibration_compare_by_vals=["branch_length", "t"],
        **kwargs,
    )
    pd.testing.assert_frame_equal(by_column, default)
    by_length = _estimate_from_masks(
        phylogeny_df, calibration_compare_by_vals=["branch_length"], **kwargs
    )
    assert not by_length.equals(default)


def test_calibration_partition_by():
    phylogeny_df, traits = _make_cherries(2_000)
    kwargs = dict(trait_masks=traits, calibrate=True)
    default = _estimate_from_masks(phylogeny_df, **kwargs)
    # one partition, as default
    one = _estimate_from_masks(
        phylogeny_df.assign(p=3), calibration_partition_by="p", **kwargs
    )
    pd.testing.assert_frame_equal(one, default)
    # partitions by cherry origin time
    by_time = _estimate_from_masks(
        phylogeny_df.assign(p=(phylogeny_df["origin_time"] > 30).astype(int)),
        calibration_partition_by="p",
        **kwargs,
    )
    assert not by_time.equals(default)
    assert np.isfinite(by_time["selection_coef"]).all()


@pytest.mark.parametrize("length_rtol", [0.0, 1e-2])
def test_length_rtol(length_rtol: float):
    phylogeny_df, traits = _make_cherries(4_000)
    kwargs = dict(trait_masks=traits, calibrate=True)
    default = _estimate_from_masks(phylogeny_df, **kwargs)
    res = _estimate_from_masks(phylogeny_df, length_rtol=length_rtol, **kwargs)
    np.testing.assert_allclose(res["gain_ratio"], default["gain_ratio"], 1e-2)
    np.testing.assert_allclose(
        res["selection_coef"], default["selection_coef"], atol=3e-4
    )


@pytest.mark.parametrize("length_rtol", [-1e-3, np.nan])
def test_invalid_length_rtol_raises(length_rtol: float):
    phylogeny_df, traits = _make_cherries(100)
    with pytest.raises(ValueError, match="length_rtol"):
        _estimate_from_masks(
            phylogeny_df, trait_masks=traits, length_rtol=length_rtol
        )


def test_calibrate_max_rate_ratio():
    phylogeny_df, traits = _make_cherries(4_000)
    kwargs = dict(
        trait_masks=traits,
        calibrate=True,
        calibrate_neighborhood_fractions=[0.1],
    )
    default = _estimate_from_masks(phylogeny_df, **kwargs)
    uncapped = _estimate_from_masks(
        phylogeny_df, calibrate_max_rate_ratio=np.inf, **kwargs
    )
    pd.testing.assert_frame_equal(default, uncapped)
    # a tight cap clips the larger calibration factors, changing the fit
    capped = _estimate_from_masks(
        phylogeny_df, calibrate_max_rate_ratio=1.0, **kwargs
    )
    assert not np.allclose(capped["gain_rate"], default["gain_rate"])


@pytest.mark.parametrize("calibrate_max_rate_ratio", [0.0, -1.0, np.nan])
def test_calibrate_max_rate_ratio_not_positive_raises(
    calibrate_max_rate_ratio: float,
):
    phylogeny_df, traits = _make_cherries(100)
    with pytest.raises(ValueError, match="calibrate_max_rate_ratio"):
        _estimate_from_masks(
            phylogeny_df,
            trait_masks=traits,
            calibrate_max_rate_ratio=calibrate_max_rate_ratio,
        )


def test_heldout_neutral_without_carriers_warns():
    phylogeny_df, traits = _make_cherries(2_000)
    with pytest.warns(UserWarning, match="does not inform"):
        _estimate_from_masks(
            phylogeny_df,
            trait_masks=[*traits, np.zeros_like(traits[0])],
            neutral_traits=[1, 2, 5],
            calibrate=True,
        )


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
        _estimate_leaves(
            df_pl,
            trait_transitions=["a", "b", "c"],
            neutral_traits=["b", "c"],
        )


def test_expression_transitions_raise():
    df_pl = _make_cherries_polars(500)
    with pytest.raises(TypeError):
        _estimate_leaves(df_pl, trait_transitions=["focal", pl.col("n1")])


@pytest.mark.parametrize(
    "df_pl",
    [
        pl.DataFrame(
            {"id": [1, 0], "ancestor_id": [0, 0], "origin_time": [0.0, 1.0]}
        ),
        pl.DataFrame(
            {"id": [0, 2], "ancestor_id": [0, 0], "origin_time": [0.0, 1.0]}
        ),
    ],
)
def test_unsupported(df_pl: pl.DataFrame):
    with pytest.raises(NotImplementedError, match="not supported"):
        _estimate_leaves(
            df_pl.with_columns(a=pl.lit(0), b=pl.lit(0), c=pl.lit(0)),
            trait_transitions=["a", "b", "c"],
            neutral_traits=["b", "c"],
        )


def test_null_transitions_raise():
    df_pl = _make_cherries_polars(2_000).with_columns(
        focal=pl.when(pl.col("id") == 5).then(None).otherwise(pl.col("focal"))
    )
    with pytest.raises(NotImplementedError, match="null"):
        _estimate_leaves(
            df_pl,
            trait_transitions=["focal", "n1", "n2"],
            neutral_traits=["n1", "n2"],
        )


def test_ancestor_list_column():
    df_pl = _make_cherries_polars(2_000)
    expected = _estimate_leaves(
        df_pl,
        trait_transitions=["focal", "n1", "n2"],
        neutral_traits=["n1", "n2"],
    )
    with_list = df_pl.with_columns(
        ancestor_list=pl.when(pl.col("id") == pl.col("ancestor_id"))
        .then(pl.lit("[none]"))
        .otherwise(pl.format("[{}]", pl.col("ancestor_id"))),
    )
    res = _estimate_leaves(
        with_list,
        trait_transitions=["focal", "n1", "n2"],
        neutral_traits=["n1", "n2"],
    )
    assert res.equals(expected)


def test_documented_unifurcation_collapse_matches():
    # as the estimator's Notes describe: sum transitions into states,
    # collapse, then remake transitions
    phylogeny_df, traits = _make_cherries(2_000)
    phylogeny_df, names = add_trait_transitions(phylogeny_df, traits[:3])
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
    knuckles[names[1:]] = 0  # neutral gains stay below the knuckle
    knuckled = phylogeny_df.copy()
    knuckled.loc[leaves, "ancestor_id"] = knuckles["id"].to_numpy()
    # each pendant branch's gain is recorded above the knuckle
    above = pd.concat(
        [knuckled.assign(**{names[0]: 0}), knuckles], ignore_index=True
    )
    df_pl = pl.from_pandas(alifestd_to_working_format(above)).drop(
        "ancestor_list", strict=False
    )
    for name in names:
        df_pl = alifestd_mark_lineage_cumsum_polars(df_pl, name, mark_as=name)
    df_pl = alifestd_assign_contiguous_ids_polars(
        alifestd_collapse_unifurcations_polars(
            df_pl, ignore_topological_sensitivity=True
        ),
    )
    for name in names:
        df_pl = alifestd_mark_lineage_delta_polars(
            df_pl, name, basal_value=False, mark_as=name
        )
    kwargs = dict(trait_transitions=names, neutral_traits=names[1:])
    res = _estimate_leaves(df_pl, **kwargs)
    expected = _estimate_leaves(
        pl.from_pandas(alifestd_to_working_format(phylogeny_df)).drop(
            "ancestor_list", strict=False
        ),
        **kwargs,
    )
    pltest.assert_frame_equal(res, expected, rel_tol=1e-9)
    assert res["num_leaf_gains"][0] == traits[0].sum()
