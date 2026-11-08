import polars as pl
import polars.selectors as cs


def tgs_assign_calibration_bins(exposure_df: pl.DataFrame) -> pl.DataFrame:
    """Implementation detail for `alifestd_tgs_pendantmdc_estimate_polars` and
    `alifestd_tgs_allmdc_estimate_polars`. Acronym tgs stands for trait
    gain/selection.

    Returns each exposure episode's calibration bin and partition, one row
    per row of `exposure_df`, numbering distinct (`calibration_partition`,
    `calibration_bin`) pairs consecutively, partition-major, so that each
    partition's bins are contiguous; see Input and Output Schema.

    Raises ValueError if `exposure_df` has no exposure episodes.

    Input Schema
    ------------
    'calibration_partition' : integer, optional
        Each episode's calibration partition id; if missing, all 0.
    'calibration_bin' : integer, optional
        Each episode's calibration bin id; if missing, all 0. So, without
        either column, every episode is in one bin.

    Output Schema
    -------------
    'exposure_id' : integer
        Id of the episode, its row index in `exposure_df`.
    'bin' : integer
        Calibration bin index, from 0, over (partition, bin) pairs, sorted.
    'partition' : integer
        Calibration partition index, from 0, over partitions, sorted.
    """
    if exposure_df.is_empty():
        raise ValueError("phylogeny_df has no exposure episodes to fit")

    calibration_df = exposure_df.select(
        exposure_id=pl.int_range(pl.len(), dtype=pl.Int64),
        # defaults, overwritten by `exposure_df`'s columns, if any
        calibration_partition=pl.lit(0),
        calibration_bin=pl.lit(0),
    ).with_columns(
        exposure_df.select(
            cs.by_name(
                "calibration_partition", "calibration_bin", require_all=False
            ),
        ),
    )
    return calibration_df.select(
        "exposure_id",
        bin=pl.struct("calibration_partition", "calibration_bin").rank("dense")
        - 1,
        partition=pl.col("calibration_partition").rank("dense") - 1,
    ).cast(pl.Int64)
