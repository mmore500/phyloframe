import logging
import typing
import warnings

import numpy as np
import polars as pl
from scipy import sparse

from ..._auxlib._calc_knn_neighborhoods import calc_knn_neighborhoods
from ._tgs_calc_heldout_nll import tgs_calc_heldout_nll


def tgs_select_calibration_neighborhoods(
    *,
    bin_coords: np.ndarray,
    neighborhood_fractions: typing.Sequence[float],
    trait_bin_counts_df: pl.DataFrame,
    neutral_traits: typing.Sequence[int],
    trait_names: typing.Sequence[str],
    max_rate_ratio: float,
) -> sparse.spmatrix:
    """Implementation detail for `alifestd_tgs_pendantmdc_estimate_polars` and
    `alifestd_tgs_allmdc_estimate_polars`. Acronym tgs stands for trait
    gain/selection.

    Returns the bin-by-bin membership matrix of calibration neighborhoods,
    the bins whose neutral trait gains calibrate each bin's branch lengths
    (see `tgs_calc_calibration_rescaling`): each bin's nearest bins by
    `bin_coords`, of the size, among `neighborhood_fractions`, that best
    predicts each held-out neutral trait's gains when calibrating against
    the other neutral traits (see `tgs_calc_heldout_nll`), given
    `trait_bin_counts_df` (see `tgs_count_trait_bin_exposures`). Warns about any
    neighborhoods without neutral gains.
    """

    def score(fraction: float) -> typing.Tuple[float, sparse.spmatrix]:
        # each bin's nearest bins, itself included, together holding about
        # `fraction` of bins
        calibration_neighborhoods = calc_knn_neighborhoods(
            bin_coords,
            num_neighbors=int(np.ceil(fraction * len(bin_coords))),
        )
        # holding out each neutral trait in turn
        nll = sum(
            tgs_calc_heldout_nll(
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
            f"{fraction:.1%} of bins give "
            f"held-out neutral negative log likelihood {nll:.1f}",
        )
        return nll, calibration_neighborhoods

    # first (smallest-index) candidate among ties; nan scores lose
    best_nll, best_neighborhoods, best_fraction = min(
        ((*score(fraction), fraction) for fraction in neighborhood_fractions),
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
        f"neighborhoods of {best_fraction:.1%} of {len(bin_coords)} "
        f"bins, {num_empty} without neutral trait gains",
    )
    if num_empty:
        warnings.warn(
            f"{num_empty} of {len(bin_coords)} calibration "
            "neighborhoods contain no neutral trait gains, so branches "
            "there get zero effective length; more neutral traits or "
            "leaves would help.",
        )
    return best_neighborhoods
