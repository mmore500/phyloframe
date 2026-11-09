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
from ._tgs_count_trait_bin_exposures import tgs_count_trait_bin_exposures


def _tgs_calc_heldout_nll(
    *,
    trait_id: int,
    trait_bin_counts_df: pl.DataFrame,
    **rescaling_kwargs,
) -> float:
    """Implementation detail for `tgs_select_calibration_neighborhoods`.

    Returns the negative log likelihood of neutral `trait_id`'s gains
    under no selection, with branch lengths calibrated against the other
    neutral traits only (see `tgs_calc_calibration_rescaling`); 0 without
    gains.
    """
    trait_counts_df = trait_bin_counts_df.filter(
        pl.col("trait_id") == trait_id
    )
    num_trials = trait_counts_df["num_trials"].to_numpy()
    num_successes = trait_counts_df["num_successes"].to_numpy()
    mean_lengths = np.divide(
        trait_counts_df["length_sum"].to_numpy(),
        num_trials,
        out=np.zeros(num_trials.shape),
        where=num_trials > 0,
    )
    calibrated_lengths = mean_lengths * tgs_calc_calibration_rescaling(
        trait_id=trait_id,
        trait_bin_counts_df=trait_bin_counts_df,
        **rescaling_kwargs,
    )
    exposure = (num_trials * calibrated_lengths).sum()
    if not (num_successes.any() and exposure):
        return 0.0  # uninformative
    gain_rate = num_successes.sum() / exposure  # fit under no selection
    return calc_binomial_nll_proxy(
        num_trials, num_successes, gain_rate * calibrated_lengths
    )[0]


def _tgs_select_partition_neighborhoods(
    bin_coords: np.ndarray,
    *,
    neighborhood_fractions: typing.Sequence[float],
    neutral_trait_ids: typing.Sequence[int],
    **heldout_kwargs,
) -> sparse.csr_matrix:
    """Implementation detail for `tgs_select_calibration_neighborhoods`.

    Returns calibration neighborhoods for one partition's bins, located at
    `bin_coords`.
    """
    candidates = [
        # each bin's nearest bins, itself included
        calc_knn_neighborhoods(
            bin_coords, num_neighbors=round(fraction * len(bin_coords))
        )
        for fraction in neighborhood_fractions
    ]
    nlls = [
        sum(  # holding out each neutral trait in turn
            _tgs_calc_heldout_nll(
                trait_id=trait_id,
                calibration_neighborhoods=calibration_neighborhoods,
                neutral_trait_ids=neutral_trait_ids,
                **heldout_kwargs,
            )
            for trait_id in neutral_trait_ids
        )
        for calibration_neighborhoods in candidates
    ]
    for fraction, nll in zip(neighborhood_fractions, nlls):
        logging.info(
            f"- tgs: calibration neighborhoods of {fraction:.1%} of "
            f"{len(bin_coords)} bins give held-out neutral negative log "
            f"likelihood {nll:.1f}",
        )
    # first candidate among ties; nan scores lose
    best = min(range(len(nlls)), key=lambda i: (np.isnan(nlls[i]), nlls[i]))
    logging.info(
        "- tgs: selected calibration neighborhoods of "
        f"{neighborhood_fractions[best]:.1%} of bins",
    )
    return candidates[best]


def tgs_select_calibration_neighborhoods(
    exposure_df: pl.DataFrame,
    *,
    neighborhood_fractions: typing.Sequence[float],
    neutral_trait_ids: typing.Sequence[int],
    max_rate_ratio: float,
) -> sparse.csr_matrix:
    """Implementation detail for `alifestd_tgs_estimate_polars`.

    Returns a sparse 0/1 bin-by-bin matrix whose row `i` marks calibration
    bin `i`'s neighborhood, i.e., the bins whose neutral trait gains
    calibrate its branch lengths (see `tgs_calc_calibration_rescaling`).

    A bin's neighborhood is its nearest bins within its partition, by its
    episodes' mean ranks of `exposure_df`'s `calibration_compare_by_*`
    columns (see `tgs_build_exposure_df_polars`). Neighborhood size is
    selected per partition, as the fraction of its bins, among
    `neighborhood_fractions`, that best predicts each neutral trait's gains
    when calibrating against the other neutral traits.
    """
    trait_bin_counts_df = tgs_count_trait_bin_exposures(
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
    calibration_neighborhoods = sparse.block_diag(
        [
            _tgs_select_partition_neighborhoods(
                partition_df.drop("partition_id", "bin_id").to_numpy(),
                neighborhood_fractions=neighborhood_fractions,
                neutral_trait_ids=neutral_trait_ids,
                # bins are numbered partition-major, so renumber from 0
                trait_bin_counts_df=trait_bin_counts_df.filter(
                    pl.col("bin_id").is_in(partition_df["bin_id"].implode())
                ).with_columns(
                    pl.col("bin_id") - partition_df["bin_id"].min()
                ),
                max_rate_ratio=max_rate_ratio,
            )
            for partition_df in bin_coords_df.partition_by(
                "partition_id", maintain_order=True
            )
        ],
        format="csr",
    )
    del trait_bin_counts_df, bin_coords_df  # clear memory
    gc.collect()
    log_memory_usage(logging.info)
    return calibration_neighborhoods
