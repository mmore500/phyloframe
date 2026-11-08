import typing

import numpy as np
import polars as pl


def tgs_assign_calibration_bins(
    exposure_df: pl.DataFrame,
) -> typing.Tuple[np.ndarray, typing.List[int]]:
    """Implementation detail for `alifestd_tgs_pendantmdc_estimate_polars` and
    `alifestd_tgs_allmdc_estimate_polars`. Acronym tgs stands for trait
    gain/selection.

    Returns each exposure episode's calibration bin, numbering the
    distinct (`calibration_partition`, `calibration_bin`) pairs of
    `exposure_df` consecutively, partition-major, so that each partition's
    bins are contiguous, and the partition bounds: each partition's first
    bin, then the number of bins. A missing column counts as all 0, so
    without either, every episode is in one bin.

    Raises ValueError if `exposure_df` has no exposure episodes.
    """
    num_episodes = len(exposure_df)
    if not num_episodes:
        raise ValueError("phylogeny_df has no exposure episodes to fit")

    pairs, bins = np.unique(
        np.column_stack(
            [
                (
                    exposure_df[column].to_numpy()
                    if column in exposure_df.columns
                    else np.zeros(num_episodes, dtype=np.int64)
                )
                for column in ("calibration_partition", "calibration_bin")
            ],
        ),
        axis=0,
        return_inverse=True,
    )
    __, partition_starts = np.unique(pairs[:, 0], return_index=True)
    return bins.ravel(), [*partition_starts.tolist(), len(pairs)]
