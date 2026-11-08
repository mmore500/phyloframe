import polars as pl
import pytest

from phyloframe.legacy._impl import tgs_assign_calibration_bins


def test_partition_major_numbering():
    exposure_df = pl.DataFrame(
        {
            "calibration_partition": [7, 3, 7, 3, 7],
            "calibration_bin": [1, 5, 0, 5, 1],
        },
    )
    res = tgs_assign_calibration_bins(exposure_df)
    # pairs (3, 5), (7, 0), (7, 1), in order
    assert res["bin"].to_list() == [2, 0, 1, 0, 2]
    assert res["partition"].to_list() == [1, 0, 1, 0, 1]
    assert res["exposure_id"].to_list() == [0, 1, 2, 3, 4]


@pytest.mark.parametrize("columns", [{}, {"calibration_bin": [4, 4, 9]}])
def test_missing_columns_count_as_zero(columns: dict):
    exposure_df = pl.DataFrame({"x": [0.0, 1.0, 2.0], **columns})
    res = tgs_assign_calibration_bins(exposure_df)
    assert res["bin"].to_list() == ([0, 0, 1] if columns else [0, 0, 0])
    assert res["partition"].to_list() == [0, 0, 0]


def test_no_episodes_raises():
    with pytest.raises(ValueError, match="no exposure episodes"):
        tgs_assign_calibration_bins(pl.DataFrame({"x": []}))
