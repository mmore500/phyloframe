import numpy as np
import polars as pl
import pytest

from phyloframe.legacy import alifestd_mark_joint_quantile_bins_polars


def _make_df(num_rows: int = 400, seed: int = 1) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    return pl.DataFrame(
        {
            "id": range(num_rows),
            "x": rng.random(num_rows),
            "y": rng.random(num_rows),
            "p": rng.integers(0, 3, num_rows),
        },
    )


@pytest.mark.parametrize("apply", [lambda x: x, lambda x: x.lazy()])
def test_bins(apply):
    res = (
        alifestd_mark_joint_quantile_bins_polars(
            apply(_make_df()), bin_by=["x", "y"], max_bins=8
        )
        .lazy()
        .collect()
    )
    assert sorted(res["joint_quantile_bin"].unique().to_list()) == list(
        range(8)
    )
    assert set(
        res["joint_quantile_bin"].value_counts()["count"].to_list()
    ) == {50}


def test_partition_by():
    df = _make_df()
    res = alifestd_mark_joint_quantile_bins_polars(
        df, bin_by=["x", "y"], partition_by="p", max_bins=4, mark_as="b"
    )
    # bins are numbered consecutively across partitions, in partition order
    assert sorted(res["b"].unique().to_list()) == list(range(12))
    by_bin = res.group_by("b").agg(
        num_partitions=pl.col("p").n_unique(), partition=pl.col("p").first()
    )
    assert (by_bin["num_partitions"] == 1).all()
    assert by_bin.sort("b")["partition"].to_list() == sorted(
        by_bin["partition"].to_list()
    )


def test_min_bin_size():
    res = alifestd_mark_joint_quantile_bins_polars(
        _make_df(40), bin_by=["x"], max_bins=1024, min_bin_size=10
    )
    assert res["joint_quantile_bin"].n_unique() == 4


def test_row_order_invariant():
    df = _make_df()
    expected = alifestd_mark_joint_quantile_bins_polars(df, bin_by=["x", "y"])
    order = np.random.default_rng(2).permutation(len(df))
    res = alifestd_mark_joint_quantile_bins_polars(
        df[order], bin_by=["x", "y"]
    )
    assert res.sort("id").equals(expected)


def test_constant_columns_ignored():
    df = _make_df().with_columns(c=pl.lit(1.0))
    expected = alifestd_mark_joint_quantile_bins_polars(df, bin_by=["x"])
    res = alifestd_mark_joint_quantile_bins_polars(df, bin_by=["c", "x"])
    assert res["joint_quantile_bin"].equals(expected["joint_quantile_bin"])


def test_empty():
    res = alifestd_mark_joint_quantile_bins_polars(
        _make_df().clear(), bin_by=["x"]
    )
    assert res["joint_quantile_bin"].to_list() == []


@pytest.mark.parametrize(
    "kwargs, error",
    [
        (dict(bin_by=[]), ValueError),
        (dict(bin_by=["absent"]), ValueError),
        (dict(bin_by=["x"], partition_by="absent"), ValueError),
        (dict(bin_by=["x"], max_bins=6), ValueError),
        (dict(bin_by=["x"], min_bin_size=0), ValueError),
        (dict(bin_by="x"), TypeError),
    ],
)
def test_invalid(kwargs, error):
    with pytest.raises(error):
        alifestd_mark_joint_quantile_bins_polars(_make_df(), **kwargs)


@pytest.mark.parametrize("missing", [None, float("nan")])
def test_missing_values_marked_negative(missing):
    df = _make_df().with_columns(
        x=pl.when(pl.col("id") == 2).then(missing).otherwise(pl.col("x"))
    )
    res = alifestd_mark_joint_quantile_bins_polars(
        df, bin_by=["x"], max_bins=8
    )
    assert res["joint_quantile_bin"][2] == -1
    expected = alifestd_mark_joint_quantile_bins_polars(
        df.filter(pl.col("id") != 2), bin_by=["x"], max_bins=8
    )
    assert res.filter(pl.col("id") != 2)["joint_quantile_bin"].equals(
        expected["joint_quantile_bin"]
    )
