import gc
import logging
import typing

import numpy as np
import polars as pl
import polars.selectors as cs
from scipy import sparse

from ..._auxlib._calc_binomial_nll_proxy import calc_binomial_nll_proxy
from ..._auxlib._calc_knn_neighborhoods import calc_knn_neighborhoods
from ..._auxlib._log_memory_usage import log_memory_usage
from ._tgs_calc_calibration_rescaling import (
    tgs_calc_calibration_rescaling,
)
from ._tgs_count_trait_exposures_binwise import (
    tgs_count_trait_exposures_binwise,
)


def _tgs_calc_heldout_nll(
    *,
    trait_id: int,
    bin_ids: np.ndarray,
    trait_bin_counts_df: pl.DataFrame,
    **calibration_rescaling_kwargs,
) -> float:
    """Implementation detail for `tgs_select_calibration_neighborhoods`.

    Returns the negative log likelihood of neutral `trait_id`'s gains in
    one partition's bins, `bin_ids`, under no selection, with episode
    durations calibrated against the other neutral traits only (see
    `tgs_calc_calibration_rescaling`); 0 without gains.
    """
    trait_counts_df = trait_bin_counts_df.filter(
        pl.col("trait_id") == trait_id,
        pl.col("bin_id").is_in(bin_ids),
    )
    num_episodes = trait_counts_df["num_episodes"].to_numpy()
    num_gains = trait_counts_df["num_gains"].to_numpy()
    mean_durations = np.divide(
        trait_counts_df["total_episode_duration"].to_numpy(),
        num_episodes,
        out=np.zeros(num_episodes.shape),
        where=num_episodes > 0,
    )
    calibrated_durations = (
        mean_durations
        * tgs_calc_calibration_rescaling(
            trait_id=trait_id,
            trait_bin_counts_df=trait_bin_counts_df,
            **calibration_rescaling_kwargs,
        )[bin_ids]
    )

    # over the partition's bins
    total_exposure = (num_episodes * calibrated_durations).sum()
    if not (num_gains.any() and total_exposure):
        return 0.0  # no gains, so every neighborhood size scores alike

    gain_rate = num_gains.sum() / total_exposure  # fit under no selection
    return calc_binomial_nll_proxy(
        num_episodes, num_gains, gain_rate * calibrated_durations
    )[0]


def _tgs_select_neighborhoods_within_partition(
    bin_coords: np.ndarray,
    *,
    bin_ids: np.ndarray,
    num_bins: int,
    neighborhood_fractions: typing.Sequence[float],
    neutral_trait_ids: typing.Sequence[int],
    **heldout_kwargs,
) -> sparse.csr_matrix:
    """Implementation detail for `tgs_select_calibration_neighborhoods`.

    Returns calibration neighborhoods for one partition's bins, `bin_ids`,
    located at `bin_coords`, as their rows of a matrix over all `num_bins`
    bins; other bins' rows are empty.
    """

    def calc_candidate_neighborhoods(
        neighborhood_fraction: float,
    ) -> sparse.csr_matrix:
        # each bin's nearest bins in the partition, itself included
        rows, cols = calc_knn_neighborhoods(
            bin_coords,
            num_neighbors=round(neighborhood_fraction * len(bin_coords)),
        ).nonzero()
        # among all bins, so gain rates compare to overall, as when fitting
        return sparse.csr_matrix(
            (np.ones(len(rows)), (bin_ids[rows], bin_ids[cols])),
            shape=(num_bins, num_bins),
        )

    candidate_neighborhoods = [
        *map(calc_candidate_neighborhoods, neighborhood_fractions)
    ]

    nlls = [
        sum(  # holding out each neutral trait in turn
            _tgs_calc_heldout_nll(
                trait_id=trait_id,
                bin_ids=bin_ids,
                calibration_neighborhoods=calibration_neighborhoods,
                neutral_trait_ids=neutral_trait_ids,
                **heldout_kwargs,
            )
            for trait_id in neutral_trait_ids
        )
        for calibration_neighborhoods in candidate_neighborhoods
    ]

    for neighborhood_fraction, nll in zip(neighborhood_fractions, nlls):
        logging.info(
            "- tgs: calibration neighborhoods of "
            f"{neighborhood_fraction:.1%} of "
            f"{len(bin_coords)} bins give held-out neutral negative log "
            f"likelihood {nll:.1f}",
        )

    # first-seen candidate wins ties; nan scores lose
    best_idx = min(
        range(len(nlls)), key=lambda i: (np.isnan(nlls[i]), nlls[i])
    )
    logging.info(
        "- tgs: selected calibration neighborhoods of "
        f"{neighborhood_fractions[best_idx]:.1%} of bins",
    )
    return candidate_neighborhoods[best_idx]


def tgs_select_calibration_neighborhoods(
    exposure_df: pl.DataFrame,
    *,
    neighborhood_fractions: typing.Sequence[float],
    neutral_trait_ids: typing.Sequence[int],
) -> sparse.csr_matrix:
    """Implementation detail for `alifestd_tgs_estimate_polars`.

    Returns a sparse 0/1 bin-by-bin matrix whose row `i` marks calibration
    bin `i`'s neighborhood, i.e., the bins whose neutral trait gains
    calibrate its episode durations (see `tgs_calc_calibration_rescaling`).

    A bin's neighborhood is its nearest bins within its partition, by its
    episodes' mean ranks of `exposure_df`'s `calibration_compare_by_*`
    columns (see `tgs_build_exposure_df_polars`). Each partition's
    neighborhood size is the fraction of its bins among
    `neighborhood_fractions` that best predicts each neutral trait's gains
    when calibrating against the other neutral traits.
    """
    trait_bin_counts_df = tgs_count_trait_exposures_binwise(
        exposure_df,
        trait_ids=neutral_trait_ids,
        weights=np.ones(len(exposure_df)),
    )

    bin_coords_df = (
        exposure_df.select(
            "partition_id",
            "bin_id",
            # tied episodes share a rank, so row order does not matter
            cs.starts_with("calibration_compare_by_").rank("average")
            / pl.len(),
        )
        .group_by("partition_id", "bin_id")
        .mean()
        .sort("bin_id")
    )
    num_bins = len(bin_coords_df)  # one row per bin

    # each partition fills its own bins' rows, combined by sum
    calibration_neighborhoods = sum(
        _tgs_select_neighborhoods_within_partition(
            partition_df.drop("partition_id", "bin_id").to_numpy(),
            bin_ids=partition_df["bin_id"].to_numpy(),
            num_bins=num_bins,
            neighborhood_fractions=neighborhood_fractions,
            neutral_trait_ids=neutral_trait_ids,
            trait_bin_counts_df=trait_bin_counts_df,
        )
        for partition_df in bin_coords_df.partition_by(
            "partition_id", maintain_order=True
        )
    )

    del trait_bin_counts_df, bin_coords_df  # free memory
    gc.collect()
    log_memory_usage(logging.info)
    return calibration_neighborhoods
