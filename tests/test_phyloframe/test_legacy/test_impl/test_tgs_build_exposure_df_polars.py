import warnings

import polars as pl
import pytest

from phyloframe.legacy._impl import tgs_build_exposure_df_polars
from phyloframe.legacy._impl._tgs_build_exposure_df_polars import (
    _tgs_cast_origin_time_float,
)


def _build(phylogeny_df, **kwargs) -> pl.DataFrame:
    return tgs_build_exposure_df_polars(
        phylogeny_df,
        **{
            "trait_transition_cols": ["t"],
            "caller": "test",
            "calibration_partition_by": None,
            "calibration_compare_by_bins": None,
            "calibration_compare_by_vals": ["branch_length"],
            "episode_mask": None,
            **kwargs,
        },
    )


def _make_tree() -> pl.DataFrame:
    #       0
    #     1   2
    #    3 4
    # trait t is gained on the branch into 1, so inherited by 3 and 4
    return pl.DataFrame(
        {
            "id": [0, 1, 2, 3, 4],
            "ancestor_id": [0, 0, 0, 1, 1],
            "origin_time": [0.0, 1.0, 2.0, 5.0, 6.0],
            "t": [0, 1, 0, 0, 0],
        },
    )


@pytest.mark.parametrize("lazy", [False, True])
def test_prepare(lazy: bool):
    df_pl = _make_tree()
    res = _build(
        df_pl.lazy() if lazy else df_pl,
        trait_transition_cols=["t"],
        caller="x",
    )
    # one row per non-root node, in order
    assert res["origin_time"].to_list() == [1.0, 2.0, 5.0, 6.0]
    assert res["branch_length"].to_list() == [1.0, 2.0, 4.0, 5.0]
    assert res["clade_size"].to_list() == [2, 1, 1, 1]
    assert res["survival_duration"].to_list() == [5.0, 0.0, 0.0, 0.0]
    # 3 and 4 inherit t from 1, so start with it
    assert res["trait0_starts_absent"].to_list() == [True, True, False, False]
    assert res["trait0_gain"].to_list() == [True, False, False, False]


def test_traits_numbered_in_order():
    res = _build(
        _make_tree().with_columns(u=pl.Series([0, 0, 1, 0, 0])),
        trait_transition_cols=["t", "u"],
        caller="x",
    )
    # t is gained on the branch into 1, u on that into leaf 2
    assert res["trait0_gain"].to_list() == [True, False, False, False]
    assert res["trait1_gain"].to_list() == [False, True, False, False]


def test_empty():
    df_pl = pl.DataFrame(
        {
            "id": pl.Series([], dtype=pl.Int64),
            "ancestor_id": pl.Series([], dtype=pl.Int64),
            "origin_time": pl.Series([], dtype=pl.Float64),
            "t": pl.Series([], dtype=pl.Int8),
        },
    )
    with pytest.raises(ValueError, match="no exposure episodes"):
        _build(df_pl)


def test_null_transitions_raise():
    df_pl = _make_tree().with_columns(
        t=pl.Series([None, 1, 0, 0, 0], dtype=pl.Int8)
    )
    with pytest.raises(NotImplementedError):
        _build(df_pl, trait_transition_cols=["t"], caller="test")


def test_missing_column_raises():
    with pytest.raises(pl.exceptions.ColumnNotFoundError):
        _build(_make_tree(), trait_transition_cols=["x"], caller="test")


def test_calibration_columns():
    df_pl = _make_tree().with_columns(
        p=pl.Series([-1, 7, 8, 9, 10]),
        b=pl.Series([-1, 3, 4, 5, 6]),
        v=pl.Series([0.0, 0.5, 1.5, 2.5, 3.5]),
        branch_length=pl.Series([9.0] * 5),
    )
    res = _build(
        df_pl,
        trait_transition_cols=["t"],
        caller="test",
        calibration_partition_by="p",
        calibration_compare_by_bins="b",
        calibration_compare_by_vals=["v", "branch_length"],
    )
    assert res["calibration_partition_id"].to_list() == [7, 8, 9, 10]
    assert res["calibration_bin_id"].to_list() == [3, 4, 5, 6]
    assert res["calibration_compare_by_v"].to_list() == [0.5, 1.5, 2.5, 3.5]
    # the exposure table's own branch lengths take precedence
    assert res["calibration_compare_by_branch_length"].to_list() == [
        1.0,
        2.0,
        4.0,
        5.0,
    ]


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(calibration_partition_by="v"),
        dict(calibration_compare_by_bins="v"),
        dict(calibration_compare_by_vals=["s"]),
    ],
)
def test_invalid_calibration_column_raises(kwargs: dict):
    df_pl = _make_tree().with_columns(
        v=pl.Series([0.0, 0.5, 1.5, 2.5, 3.5]), s=pl.Series(["a"] * 5)
    )
    with pytest.raises(NotImplementedError, match="calibration column"):
        _build(df_pl, trait_transition_cols=["t"], caller="test", **kwargs)


def test_missing_calibration_column_raises():
    with pytest.raises(
        pl.exceptions.ColumnNotFoundError, match="calibration_compare_by_bins"
    ):
        _build(
            _make_tree(),
            trait_transition_cols=["t"],
            caller="test",
            calibration_compare_by_bins="x",
        )


def test_origin_time_delta_branch_lengths_match():
    # as the estimator's Notes compute branch lengths for calibration bins
    from phyloframe.legacy import alifestd_mark_origin_time_delta_polars

    df_pl = _make_tree()
    res = _build(df_pl, trait_transition_cols=["t"], caller="test")
    marked = alifestd_mark_origin_time_delta_polars(
        df_pl, mark_as="branch_length"
    )
    assert (
        marked["branch_length"][1:].to_list() == res["branch_length"].to_list()
    )


def test_unifurcation_raises():
    #     0
    #   1   2
    #   3
    # node 1 has one child
    df_pl = pl.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "ancestor_id": [0, 0, 0, 1],
            "origin_time": [0.0, 1.0, 2.0, 5.0],
            "t": [0, 1, 0, 0],
        },
    )
    with pytest.raises(NotImplementedError, match="unifurcations"):
        _build(df_pl)


def test_unifurcating_root_supported():
    # root 0 has one child, 1, which branches into leaves 2 and 3
    df_pl = pl.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "ancestor_id": [0, 0, 1, 1],
            "origin_time": [0.0, 1.0, 2.0, 4.0],
            "t": [0, 0, 1, 0],
        },
    )
    res = _build(df_pl)
    assert res["branch_length"].to_list() == [1.0, 1.0, 3.0]
    assert res["trait0_gain"].to_list() == [False, True, False]


def test_masked_out_zero_length_branch_ok():
    # leaf 2 originates with its parent, root 0, so its branch has length 0
    df_pl = _make_tree().with_columns(
        origin_time=pl.Series([0.0, 1.0, 0.0, 5.0, 6.0]),
        keep=pl.Series([True, True, False, True, True]),
    )
    with pytest.raises(ValueError, match="branch lengths must be positive"):
        _build(df_pl)
    res = _build(df_pl, episode_mask="keep")
    assert res["branch_length"].to_list() == [1.0, 4.0, 5.0]


def test_masked_out_null_calibration_value_ok():
    # leaf-only calibration bins, null at inner node 1
    df_pl = _make_tree().with_columns(
        b=pl.Series([None, None, 3, 4, 5]),
        is_leaf=pl.Series([False, False, True, True, True]),
    )
    with pytest.raises(NotImplementedError, match="calibration column"):
        _build(df_pl, calibration_compare_by_bins="b")
    res = _build(
        df_pl, calibration_compare_by_bins="b", episode_mask="is_leaf"
    )
    assert res["calibration_bin_id"].to_list() == [3, 4, 5]


def test_episode_mask_null_at_root_ignored():
    df_pl = _make_tree().with_columns(
        keep=pl.Series([None, True, True, True, True], dtype=pl.Boolean),
    )
    assert _build(df_pl, episode_mask="keep").equals(_build(df_pl))


def test_episode_mask_keeps_whole_tree_columns():
    # masking leaf 3, under kept episode 1, leaves 1's clade size and
    # survival duration as unmasked
    df_pl = _make_tree().with_columns(
        keep=pl.Series([True, True, True, False, True]),
    )
    expected = _build(df_pl).filter(pl.Series([True, True, False, True]))
    assert _build(df_pl, episode_mask="keep").equals(expected)


def test_masked_out_leaf_counts_for_survival():
    # leaf 3 is masked out, but still ancestor 1's latest leaf
    df_pl = _make_tree().with_columns(
        origin_time=pl.Series([0.0, 1.0, 2.0, 100.0, 6.0]),
        keep=pl.Series([True, True, True, False, True]),
    )
    res = _build(df_pl, episode_mask="keep")
    assert res["survival_duration"].to_list() == [99.0, 0.0, 0.0]
    assert res["clade_size"].to_list() == [2, 1, 1]


def test_nan_origin_time_skipped_for_survival():
    # leaf 3's time is unknown, so ancestor 1's latest known leaf is leaf 4
    df_pl = _make_tree().with_columns(
        origin_time=pl.Series([0.0, 1.0, 2.0, float("nan"), 6.0]),
        keep=pl.Series([True, True, True, False, True]),
    )
    res = _build(df_pl, episode_mask="keep")
    assert res["survival_duration"].to_list() == [5.0, 0.0, 0.0]
    assert res["clade_size"].to_list() == [2, 1, 1]


def test_masked_out_descendants_count_for_survival():
    df_pl = _make_tree().with_columns(
        keep=pl.Series([True, True, True, False, False]),
    )
    res = _build(df_pl, episode_mask="keep")
    assert res["survival_duration"].to_list() == [5.0, 0.0]
    assert res["clade_size"].to_list() == [2, 1]


def test_infinite_origin_time_raises():
    # kept leaf 4 with infinite time has infinite branch length
    df_pl = _make_tree().with_columns(
        origin_time=pl.Series([0.0, 1.0, 2.0, 5.0, float("inf")]),
    )
    with pytest.raises(ValueError, match="positive and finite"):
        _build(df_pl)
    # masked out, it gives its kept ancestor 1 infinite survival duration
    df_pl = df_pl.with_columns(keep=pl.Series([True, True, True, True, False]))
    with pytest.raises(ValueError, match="survival durations must be finite"):
        _build(df_pl, episode_mask="keep")


def test_nan_root_origin_time_ok_with_children_masked_out():
    df_pl = _make_tree().with_columns(
        origin_time=pl.Series([float("nan"), 1.0, 2.0, 5.0, 6.0]),
        keep=pl.Series([True, False, False, True, True]),
    )
    res = _build(df_pl, episode_mask="keep")
    assert res["branch_length"].to_list() == [4.0, 5.0]


def test_missing_episode_mask_raises_early():
    with pytest.raises(
        pl.exceptions.ColumnNotFoundError, match="episode_mask"
    ):
        _build(_make_tree(), episode_mask="keep")


def test_integer_origin_time_exact_is_quiet():
    df_pl = _make_tree().with_columns(pl.col("origin_time").cast(pl.Int64))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = _build(df_pl)
    assert res["branch_length"].to_list() == [1.0, 2.0, 4.0, 5.0]


def test_integer_origin_time_precision_loss_warns():
    with pytest.warns(UserWarning, match="precision"):
        res = _tgs_cast_origin_time_float(pl.Series([0, 2**53 + 1]))
    assert res.dtype == pl.Float64


def test_null_root_origin_time_is_unknown():
    # the root's unknown time leaves its children's branch lengths undefined
    df_pl = _make_tree().with_columns(
        origin_time=pl.Series([None, 1.0, 2.0, 5.0, 6.0])
    )
    with pytest.raises(ValueError, match="undefined"):
        _build(df_pl)
    res = _build(
        df_pl.with_columns(keep=pl.Series([True, False, False, True, True])),
        episode_mask="keep",
    )
    assert res["branch_length"].to_list() == [4.0, 5.0]
