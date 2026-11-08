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
    bins, partition_bounds = tgs_assign_calibration_bins(exposure_df)
    # pairs (3, 5), (7, 0), (7, 1), in order
    assert bins.tolist() == [2, 0, 1, 0, 2]
    assert partition_bounds == [0, 1, 3]


@pytest.mark.parametrize("columns", [{}, {"calibration_bin": [4, 4, 9]}])
def test_missing_columns_count_as_zero(columns: dict):
    exposure_df = pl.DataFrame({"x": [0.0, 1.0, 2.0], **columns})
    bins, partition_bounds = tgs_assign_calibration_bins(exposure_df)
    assert bins.tolist() == ([0, 0, 1] if columns else [0, 0, 0])
    assert partition_bounds == [0, 2 if columns else 1]


def test_no_episodes_raises():
    with pytest.raises(ValueError, match="no exposure episodes"):
        tgs_assign_calibration_bins(pl.DataFrame({"x": []}))
