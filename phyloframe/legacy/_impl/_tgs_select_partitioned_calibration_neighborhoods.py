import gc
import logging
import typing
import warnings

import numpy as np
import polars as pl
import polars.selectors as cs
from scipy import sparse

from ..._auxlib._calc_binomial_nll_proxy import calc_binomial_nll_proxy
from ..._auxlib._calc_group_means import calc_group_means
from ..._auxlib._calc_knn_neighborhoods import calc_knn_neighborhoods
from ..._auxlib._ceil_decimal_product import ceil_decimal_product
from ..._auxlib._log_memory_usage import log_memory_usage
from ._tgs_calc_calibration_rescaling import (
    tgs_calc_calibration_rescaling,
)
from ._tgs_count_trait_bin_exposures import tgs_count_trait_bin_exposures


def _tgs_calc_heldout_nll(
    *,
    calibration_neighborhoods: sparse.spmatrix,
    trait_bin_counts_df: pl.DataFrame,
    trait_id: int,
    trait_name: str,
    neutral_traits: typing.Sequence[int],
    max_rate_ratio: float,
) -> float:
    """Implementation detail for `_tgs_select_calibration_neighborhoods`.
    Acronym tgs stands for trait gain/selection.

    Returns the negative log likelihood of held-out neutral `trait_id`'s gains
    under no selection, with each bin's mean branch length calibrated
    against the other neutral traits only; 0 if it has no gains or no
    calibrated exposure, so scores the same for every neighborhood size.
    """
    trait_counts_df = trait_bin_counts_df.filter(
        pl.col("trait_id") == trait_id
    )
    num_trials = trait_counts_df["num_trials"].to_numpy()
    num_successes = trait_counts_df["num_successes"].to_numpy()
    mean_bin_branch_lengths = np.divide(
        trait_counts_df["length_sum"].to_numpy(),
        num_trials,
        out=np.zeros(num_trials.shape),
        where=num_trials > 0,
    )
    # held out from its own calibration, against the other neutral traits
    assert trait_id in neutral_traits
    calibrated_lengths = (
        mean_bin_branch_lengths
        * tgs_calc_calibration_rescaling(
            trait_bin_counts_df=trait_bin_counts_df,
            trait_id=trait_id,
            neutral_traits=neutral_traits,
            calibration_neighborhoods=calibration_neighborhoods,
            max_rate_ratio=max_rate_ratio,
        )
    )
    exposure = (num_trials * calibrated_lengths).sum()
    if not (num_successes.any() and exposure):
        warnings.warn(
            f"neutral trait {trait_name} has no gains or no calibrated "
            "exposure, so it does not inform the choice of calibration "
            "neighborhood size.",
        )
        return 0.0
    gain_rate = num_successes.sum() / exposure  # neutral fit, to first order
    return calc_binomial_nll_proxy(
        num_trials, num_successes, gain_rate * calibrated_lengths
    )[0]


def _tgs_select_calibration_neighborhoods(
    *,
    bin_coords: np.ndarray,
    neighborhood_fractions: typing.Sequence[float],
    trait_bin_counts_df: pl.DataFrame,
    neutral_traits: typing.Sequence[int],
    trait_names: typing.Sequence[str],
    max_rate_ratio: float,
) -> sparse.spmatrix:
    """Implementation detail for `alifestd_tgs_estimate_polars`. Acronym tgs
    stands for trait gain/selection.

    Returns the bin-by-bin membership matrix of calibration neighborhoods,
    the bins whose neutral trait gains calibrate each bin's branch lengths
    (see `tgs_calc_calibration_rescaling`): each bin's nearest bins by
    `bin_coords`, of the size, among `neighborhood_fractions`, that best
    predicts each held-out neutral trait's gains when calibrating against
    the other neutral traits (see `_tgs_calc_heldout_nll`), given
    `trait_bin_counts_df` (see `tgs_count_trait_bin_exposures`). Warns about any
    neighborhoods without neutral gains.
    """

    def calc_heldout_nll_for_neighborhood_fraction(
        neighborhood_fraction: float,
    ) -> typing.Tuple[float, sparse.spmatrix]:
        # each bin's nearest bins, itself included, together holding about
        # `neighborhood_fraction` of bins
        calibration_neighborhoods = calc_knn_neighborhoods(
            bin_coords,
            num_neighbors=ceil_decimal_product(
                neighborhood_fraction, len(bin_coords)
            ),
        )
        # holding out each neutral trait in turn
        nll = sum(
            _tgs_calc_heldout_nll(
                calibration_neighborhoods=calibration_neighborhoods,
                trait_bin_counts_df=trait_bin_counts_df,
                trait_id=trait_id,
                trait_name=trait_names[trait_id],
                neutral_traits=neutral_traits,
                max_rate_ratio=max_rate_ratio,
            )
            for trait_id in neutral_traits
        )
        logging.info(
            "- tgs: calibration neighborhoods of "
            f"{neighborhood_fraction:.1%} of bins give "
            f"held-out neutral negative log likelihood {nll:.1f}",
        )
        return nll, calibration_neighborhoods

    # first (smallest-index) candidate among ties; nan scores lose
    best_nll, best_neighborhoods, best_neighborhood_fraction = min(
        (
            (
                *calc_heldout_nll_for_neighborhood_fraction(
                    neighborhood_fraction
                ),
                neighborhood_fraction,
            )
            for neighborhood_fraction in neighborhood_fractions
        ),
        key=lambda candidate: (np.isnan(candidate[0]), candidate[0]),
    )

    neutral_gains = (
        trait_bin_counts_df.filter(pl.col("trait_id").is_in(neutral_traits))
        .group_by("bin")
        .agg(pl.col("num_successes").sum())
        .sort("bin")["num_successes"]
        .to_numpy()
    )
    num_empty = int((best_neighborhoods @ neutral_gains == 0).sum())
    logging.info(
        "- tgs: selected calibration "
        f"neighborhoods of {best_neighborhood_fraction:.1%} of {len(bin_coords)} "
        f"bins, {num_empty} without neutral trait gains",
    )
    if num_empty:
        warnings.warn(
            f"{num_empty} of {len(bin_coords)} calibration neighborhoods "
            "contain no neutral trait gains, so branches there are left "
            "uncalibrated; more neutral traits or leaves may help.",
        )
    return best_neighborhoods


def _tgs_calc_calibration_bin_coords(
    exposure_df: pl.DataFrame, *, bins: np.ndarray, num_bins: int
) -> np.ndarray:
    """Implementation detail for
    `tgs_select_partitioned_calibration_neighborhoods`.

    Returns each calibration bin's coordinates for nearest-neighbor
    comparison: its episodes' mean ranks, scaled to (0, 1], by each
    `calibration_compare_by_*` column of `exposure_df`. Tied episodes share
    their average rank, so coordinates do not depend on row order, and a
    constant column places every bin alike, adding no distance.
    """
    episode_ranks = exposure_df.select(
        cs.starts_with("calibration_compare_by_").rank("average")
    ).to_numpy()
    return calc_group_means(
        bins, episode_ranks / len(exposure_df), num_groups=num_bins
    )


def _tgs_slice_partition_bins(
    trait_bin_counts_df: pl.DataFrame, begin: int, end: int
) -> pl.DataFrame:
    """Implementation detail for
    `tgs_select_partitioned_calibration_neighborhoods`.

    Returns the rows of `trait_bin_counts_df`, per-trait exposure counts in
    each calibration bin (see `tgs_count_trait_bin_exposures`), for one partition's
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
    """Implementation detail for `alifestd_tgs_estimate_polars`. Acronym tgs
    stands for trait gain/selection.

    Selects calibration neighborhoods separately within each partition
    (see `_tgs_select_calibration_neighborhoods`), so that neighborhoods
    never span partitions. Returns a sparse 0/1 bin-by-bin matrix whose row
    `i` marks the calibration bins in bin `i`'s neighborhood, i.e., those
    whose neutral trait gains calibrate bin `i`'s branch lengths; with one
    bin, that bin alone, i.e., no calibration. Also returns per-(trait,
    bin) exposure counts (see `tgs_count_trait_bin_exposures`), which it needs for
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
    `_tgs_assign_calibration_bins`):

    'bin' : integer
        Calibration bin index, partition-major.
    'partition' : integer
        Calibration partition index.

    Output Schema
    -------------
    Per-(trait, bin) counts, as for `tgs_count_trait_bin_exposures`:

    'trait_id' : integer
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
    trait_bin_counts_df = tgs_count_trait_bin_exposures(
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
            _tgs_select_calibration_neighborhoods(
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
