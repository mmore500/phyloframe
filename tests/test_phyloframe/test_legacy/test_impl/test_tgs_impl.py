import numpy as np
import polars as pl
import polars.testing as pltest
import pytest

from phyloframe._auxlib import identity
from phyloframe.legacy import alifestd_mark_joint_quantile_bins_polars
from phyloframe.legacy._impl import (
    tgs_calc_exposure_df,
    tgs_count_episodes,
    tgs_impl,
)

from ._simulate_allmdc_exposure_df import simulate_allmdc_exposure_df

_TRAIT_NAMES = ["focal", "a", "b", "c"]


def _impl(exposure_df: pl.DataFrame, **kwargs) -> pl.DataFrame:
    return tgs_impl(
        exposure_df,
        **{
            **dict(
                counts_df=tgs_count_episodes(
                    trait_names=_TRAIT_NAMES, exposure_df=exposure_df
                ),
                trait_names=_TRAIT_NAMES,
                neutral_traits=[1, 2, 3],
                calibrate_neighborhood_fractions=(0.25, 1.0),
                calibrate_max_rate_ratio=np.inf,
                length_rtol=1e-3,
                num_bootstrap=0,
                loss_rate=0.0,
                selection_coef_starts=(0.0, -0.3, -1.8),
                growth_exponent_bound=600.0,
                seed=1,
                parallel_backend=None,
                progress_wrap=identity,
            ),
            **kwargs,
        },
    )


def _binned_exposure_df() -> pl.DataFrame:
    return alifestd_mark_joint_quantile_bins_polars(
        simulate_allmdc_exposure_df(8_000, num_neutral=3),
        bin_by=["branch_length", "origin_time"],
        max_bins=16,
        mark_as="calibration_bin",
    ).with_columns(
        calibration_partition=(pl.col("clade_size") > 1).cast(pl.Int64),
        calibration_compare_by_branch_length=pl.col("branch_length"),
        calibration_compare_by_origin_time=pl.col("origin_time"),
    )


def test_uncalibrated():
    exposure_df = simulate_allmdc_exposure_df(8_000, num_neutral=3)
    res = _impl(exposure_df)
    assert res["trait"].to_list() == _TRAIT_NAMES
    assert res["gain_ratio"][0] == pytest.approx(2.0, rel=0.3)
    # a constant partition column, like none, gives one bin
    pltest.assert_frame_equal(
        res, _impl(exposure_df.with_columns(calibration_partition=0))
    )


def test_calibrated():
    exposure_df = _binned_exposure_df()
    res = _impl(exposure_df, num_bootstrap=2)
    assert res["gain_ratio"][0] == pytest.approx(2.0, rel=0.3)
    assert "selection_coef_se" in res.columns


def test_one_neighborhood_per_partition_ignores_bins():
    # neighborhoods of all of each partition's bins calibrate as if each
    # partition were one bin
    exposure_df = _binned_exposure_df()
    whole = _impl(exposure_df, calibrate_neighborhood_fractions=(1.0,))
    by_partition = _impl(exposure_df.drop("calibration_bin"))
    pltest.assert_frame_equal(whole, by_partition, rel_tol=1e-4)


def test_partitions_separate_neighborhoods():
    # renumbering partitions so bins are shared across them changes nothing
    exposure_df = _binned_exposure_df()
    res = _impl(exposure_df)
    shifted = exposure_df.with_columns(
        calibration_partition=pl.col("calibration_partition") * 5 - 3
    )
    pltest.assert_frame_equal(res, _impl(shifted))
    # calibrating across partitions differs
    assert not np.allclose(
        res["selection_coef"],
        _impl(exposure_df.drop("calibration_partition"))["selection_coef"],
    )


def test_no_episodes_raises():
    exposure_df = tgs_calc_exposure_df(
        ancestor_ids=np.array([0]),
        origin_time=np.array([0.0]),
        trait_states=[np.array([0])] * 4,
    )
    with pytest.raises(ValueError, match="no exposure episodes"):
        _impl(exposure_df)
