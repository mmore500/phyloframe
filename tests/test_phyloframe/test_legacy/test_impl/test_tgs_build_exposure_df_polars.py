import pandas as pd
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
    by_time = res.set_index("origin_time")
    assert sorted(by_time.index) == [2.0, 5.0]
    assert by_time.loc[5.0, "branch_length"] == 5.0
    assert by_time.loc[2.0, "branch_length"] == 2.0
    assert by_time["clade_size"].tolist() == [1, 1]
    assert by_time["survival_duration"].tolist() == [0.0, 0.0]
    assert by_time["trait0_starts_absent"].all()
    assert by_time.loc[5.0, "trait0_gain"]
    assert not by_time.loc[2.0, "trait0_gain"]


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
    assert isinstance(res, pd.DataFrame)
    assert res.empty
    assert {"branch_length", "trait0_starts_absent", "trait0_gain"} <= set(res)


def test_null_transitions_raise():
    df_pl = _make_tree().with_columns(
        t=pl.Series([None, 1, 0, 0], dtype=pl.Int8)
    )
    with pytest.raises(NotImplementedError):
        tgs_build_exposure_df_polars(
            df_pl, trait_transitions=["t"], caller="test"
        )


def test_missing_column_raises():
    with pytest.raises(ValueError):
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
    by_time = res.set_index("origin_time")
    assert by_time["node_id"].tolist() == [2, 3]
    assert by_time["calibration_partition"].tolist() == [7, 8]
    assert by_time["calibration_bin"].tolist() == [3, 4]
    assert by_time["calibration_compare_by_v"].tolist() == [0.5, 1.5]
    # the exposure table's own branch lengths take precedence
    assert by_time["calibration_compare_by_branch_length"].tolist() == [
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
    with pytest.raises(ValueError):
        tgs_build_exposure_df_polars(
            _make_tree(),
            trait_transitions=["t"],
            caller="test",
            calibration_compare_by_bins="x",
        )
