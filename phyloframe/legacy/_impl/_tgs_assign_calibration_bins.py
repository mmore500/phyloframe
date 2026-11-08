import polars as pl


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
    'bin' : integer
        Calibration bin index, from 0, over (partition, bin) pairs, sorted.
    'partition' : integer
        Calibration partition index, from 0, over partitions, sorted.
    """
    if exposure_df.is_empty():
        raise ValueError("phylogeny_df has no exposure episodes to fit")

    pairs = pl.struct(
        *(
            (
                pl.col(column)
                if column in exposure_df.columns
                else pl.repeat(0, pl.len()).alias(column)
            )
            for column in ("calibration_partition", "calibration_bin")
        ),
    )
    return exposure_df.select(
        bin=pairs.rank("dense") - 1,
        partition=pairs.struct[0].rank("dense") - 1,
    ).cast(pl.Int64)
