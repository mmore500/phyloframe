import gc
import logging
import typing

import numpy as np
import polars as pl
import polars.selectors as cs
from scipy import sparse

from ..._auxlib._calc_group_means import calc_group_means
from ..._auxlib._calc_tiebroken_ranks import calc_tiebroken_ranks
from ..._auxlib._log_memory_usage import log_memory_usage
from ._tgs_count_traits import tgs_count_traits
from ._tgs_select_calibration_neighborhoods import (
    tgs_select_calibration_neighborhoods,
)


def _tgs_calc_calibration_bin_coords(
    exposure_df: pl.DataFrame, *, bins: np.ndarray, num_bins: int
) -> np.ndarray:
    """Implementation detail for
    `tgs_select_partitioned_calibration_neighborhoods`.

    Returns each calibration bin's coordinates for nearest-neighbor
    comparison: its episodes' mean ranks, scaled to (0, 1], by each
    `calibration_compare_by_*` column of `exposure_df`; constant columns
    carry no distance, so are dropped.
    """
    compare_by_df = exposure_df.select(
        cs.starts_with("calibration_compare_by_")
    )
    episode_ranks = calc_tiebroken_ranks(
        [
            compare_by_df[column].to_numpy()
            for column in compare_by_df.columns
            if compare_by_df[column].n_unique() > 1
        ]
        or [compare_by_df[compare_by_df.columns[0]].to_numpy()]
    )
    return calc_group_means(
        bins, episode_ranks / len(exposure_df), num_groups=num_bins
    )


def _tgs_slice_partition_bins(
    trait_bin_counts_df: pl.DataFrame, begin: int, end: int
) -> pl.DataFrame:
    """Implementation detail for
    `tgs_select_partitioned_calibration_neighborhoods`.

    Returns the rows of `trait_bin_counts_df`, per-trait exposure counts in
    each calibration bin (see `tgs_count_traits`), for one partition's
    bins, `begin` up to `end`, with bins renumbered from 0 to match that
    partition's own bin coordinates.
    """
    return trait_bin_counts_df.filter(
        pl.col("bin").is_between(begin, end - 1)
    ).with_columns(pl.col("bin") - begin)


def tgs_select_partitioned_calibration_neighborhoods(
    exposure_df: pl.DataFrame,
    *,
    bins_df: pl.DataFrame,
    num_bins: int,
    neighborhood_fractions: typing.Sequence[float],
    neutral_traits: typing.Sequence[int],
    trait_names: typing.Sequence[str],
    max_rate_ratio: float,
) -> typing.Tuple[sparse.csr_matrix, pl.DataFrame]:
    """Implementation detail for `alifestd_tgs_pendantmdc_estimate_polars` and
    `alifestd_tgs_allmdc_estimate_polars`. Acronym tgs stands for trait
    gain/selection.

    Selects calibration neighborhoods separately within each partition
    (see `tgs_select_calibration_neighborhoods`), so that neighborhoods
    never span partitions. Returns a sparse 0/1 bin-by-bin matrix whose row
    `i` marks the calibration bins in bin `i`'s neighborhood, i.e., those
    whose neutral trait gains calibrate bin `i`'s branch lengths; with one
    bin, that bin alone, i.e., no calibration. Also returns per-(trait,
    bin) exposure counts (see `tgs_count_traits`), which it needs for
    selection, for reuse by the point estimate; see Input and Output
    Schema.

    Input Schema
    ------------
    `exposure_df` columns (see `tgs_build_exposure_df_polars`):

    'trait{i}_starts_absent', 'trait{i}_gain' : boolean, per trait `i`
        Whether the episode started without, and whether it gained, the
        trait.
    'branch_length' : float
        Episode length.
    'calibration_compare_by_{name}' : numeric
        Values by which calibration bins are compared.

    `bins_df` columns, one row per episode (see
    `tgs_assign_calibration_bins`):

    'bin' : integer
        Calibration bin index, partition-major.
    'partition' : integer
        Calibration partition index.

    Output Schema
    -------------
    Per-(trait, bin) counts, as for `tgs_count_traits`:

    'trait' : integer
        Trait index.
    'bin' : integer
        Calibration bin index.
    'num_trials' : float
        Exposure episodes of the trait in the bin.
    'num_successes' : float
        Of those, the number that gained the trait.
    'length_sum' : float
        Their summed branch length.
    """
    bins = bins_df["bin"].to_numpy()
    trait_bin_counts_df = tgs_count_traits(
        exposure_df=exposure_df,
        bins=bins,
        num_bins=num_bins,
        weights=np.ones(len(exposure_df)),
    )
    if num_bins == 1:  # one neighborhood of all episodes, i.e., no calibration
        return sparse.identity(1, format="csr"), trait_bin_counts_df

    # each partition's bins, contiguous as numbered partition-major
    partition_bin_ranges = (
        bins_df.group_by("partition")
        .agg(begin=pl.col("bin").min(), end=pl.col("bin").max() + 1)
        .sort("partition")
        .select("begin", "end")
        .iter_rows()
    )
    bin_coords = _tgs_calc_calibration_bin_coords(
        exposure_df, bins=bins, num_bins=num_bins
    )
    calibration_neighborhoods = sparse.block_diag(
        [
            tgs_select_calibration_neighborhoods(
                bin_coords=bin_coords[begin:end],
                neighborhood_fractions=neighborhood_fractions,
                trait_bin_counts_df=_tgs_slice_partition_bins(
                    trait_bin_counts_df, begin, end
                ),
                neutral_traits=neutral_traits,
                trait_names=trait_names,
                max_rate_ratio=max_rate_ratio,
            )
            for begin, end in partition_bin_ranges
        ],
        format="csr",
    )
    gc.collect()
    log_memory_usage(logging.info)
    return calibration_neighborhoods, trait_bin_counts_df
