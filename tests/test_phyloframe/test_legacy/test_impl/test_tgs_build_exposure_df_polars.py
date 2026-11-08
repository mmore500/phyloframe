import polars as pl
import pytest

from phyloframe.legacy._impl import tgs_build_exposure_df_polars


def _make_tree() -> pl.DataFrame:
    #     0
    #   1   2
    #   3
    # node 1 is a unifurcation; trait t is gained on the branch into 1
    return pl.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "ancestor_id": [0, 0, 0, 1],
            "origin_time": [0.0, 1.0, 2.0, 5.0],
            "t": [0, 1, 0, 0],
        },
    )


@pytest.mark.parametrize("lazy", [False, True])
def test_prepare(lazy: bool):
    df_pl = _make_tree()
    res = tgs_build_exposure_df_polars(
        df_pl.lazy() if lazy else df_pl, trait_transitions=["t"], caller="x"
    )
    # unifurcation 1 is bypassed, so leaf 3 hangs from root 0, and the
    # gain into 1 is credited to the merged branch
    by_time = res.sort("origin_time")
    assert by_time["origin_time"].to_list() == [2.0, 5.0]
    assert by_time["branch_length"].to_list() == [2.0, 5.0]
    assert by_time["clade_size"].to_list() == [1, 1]
    assert by_time["survival_duration"].to_list() == [0.0, 0.0]
    assert by_time["trait0_starts_absent"].all()
    assert by_time["trait0_gain"].to_list() == [False, True]


def test_traits_numbered_in_order():
    res = tgs_build_exposure_df_polars(
        _make_tree().with_columns(u=pl.Series([1, 0, 0, 0])),
        trait_transitions=["t", "u"],
        caller="x",
    )
    # u is present from the root, so never starts absent
    assert not res["trait1_starts_absent"].any()
    assert res["trait0_gain"].any()


def test_empty():
    df_pl = pl.DataFrame(
        {
            "id": pl.Series([], dtype=pl.Int64),
            "ancestor_id": pl.Series([], dtype=pl.Int64),
            "origin_time": pl.Series([], dtype=pl.Float64),
            "t": pl.Series([], dtype=pl.Int8),
        },
    )
    res = tgs_build_exposure_df_polars(
        df_pl, trait_transitions=["t"], caller="test"
    )
    assert isinstance(res, pl.DataFrame)
    assert res.is_empty()
    assert {"branch_length", "trait0_starts_absent", "trait0_gain"} <= set(
        res.columns
    )


def test_null_transitions_raise():
    df_pl = _make_tree().with_columns(
        t=pl.Series([None, 1, 0, 0], dtype=pl.Int8)
    )
    with pytest.raises(NotImplementedError):
        tgs_build_exposure_df_polars(
            df_pl, trait_transitions=["t"], caller="test"
        )


def test_missing_column_raises():
    with pytest.raises(pl.exceptions.ColumnNotFoundError):
        tgs_build_exposure_df_polars(
            _make_tree(), trait_transitions=["x"], caller="test"
        )


def test_calibration_columns():
    df_pl = _make_tree().with_columns(
        p=pl.Series([-1, -1, 7, 8]),
        b=pl.Series([-1, -1, 3, 4]),
        v=pl.Series([0.0, 0.0, 0.5, 1.5]),
        branch_length=pl.Series([9.0] * 4),
    )
    res = tgs_build_exposure_df_polars(
        df_pl,
        trait_transitions=["t"],
        caller="test",
        calibration_partition_by="p",
        calibration_compare_by_bins="b",
        calibration_compare_by_vals=["v", "branch_length"],
    )
    by_time = res.sort("origin_time")
    assert by_time["node_id"].to_list() == [2, 3]
    assert by_time["calibration_partition"].to_list() == [7, 8]
    assert by_time["calibration_bin"].to_list() == [3, 4]
    assert by_time["calibration_compare_by_v"].to_list() == [0.5, 1.5]
    # the exposure table's own branch lengths take precedence
    assert by_time["calibration_compare_by_branch_length"].to_list() == [
        2.0,
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
        v=pl.Series([0.0, 0.0, 0.5, 1.5]), s=pl.Series(["a"] * 4)
    )
    with pytest.raises(NotImplementedError, match="calibration column"):
        tgs_build_exposure_df_polars(
            df_pl, trait_transitions=["t"], caller="test", **kwargs
        )


def test_missing_calibration_column_raises():
    with pytest.raises(pl.exceptions.ColumnNotFoundError):
        tgs_build_exposure_df_polars(
            _make_tree(),
            trait_transitions=["t"],
            caller="test",
            calibration_compare_by_bins="x",
        )


def test_stem_node_branch_lengths_match():
    # as the estimators' Notes compute branch lengths for calibration bins
    from phyloframe.legacy import alifestd_mark_stem_node_id_polars

    df_pl = _make_tree()
    res = tgs_build_exposure_df_polars(
        df_pl, trait_transitions=["t"], caller="test"
    )
    marked = alifestd_mark_stem_node_id_polars(df_pl).with_columns(
        branch_length=pl.col("origin_time")
        - pl.col("origin_time").gather(pl.col("stem_node_id")),
    )
    assert (
        marked["branch_length"].gather(res["node_id"].to_numpy()).to_list()
        == res["branch_length"].to_list()
    )
