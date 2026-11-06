import logging
import typing
import warnings

import numpy as np
import pandas as pd
from scipy import sparse

from ..._auxlib._calc_knn_neighborhoods import calc_knn_neighborhoods
from ._tgs_calc_heldout_nll import tgs_calc_heldout_nll


def tgs_select_calibration_neighborhoods(
    *,
    bin_coords: np.ndarray,
    neighborhood_fractions: typing.Sequence[float],
    counts: pd.DataFrame,
    neutral_traits: typing.Sequence[int],
    trait_names: typing.Sequence[str],
    max_rate_ratio: float,
) -> sparse.spmatrix:
    """Implementation detail for `alifestd_tgs_pendantmdc_estimate_polars` and
    `alifestd_tgs_allmdc_estimate_polars`. Acronym tgs stands for trait
    gain/selection.

    Returns the bin-by-bin membership matrix of calibration neighborhoods,
    the bins whose neutral trait gains calibrate each bin's branch lengths
    (see `tgs_calc_calibration_factors`): each bin's nearest bins by
    `bin_coords`, of the size, among `neighborhood_fractions`, that best
    predicts each held-out neutral trait's gains when calibrating against
    the other neutral traits (see `tgs_calc_heldout_nll`), given (`trait`, `bin`)-indexed `counts` (see
    `tgs_count_traits`). Warns about any neighborhoods without neutral gains.
    """
    best_nll, best_neighborhoods, best_fraction = np.inf, None, np.nan
    for fraction in neighborhood_fractions:
        # each bin's nearest bins, itself included, together
        # holding about `fraction` of bins
        calibration_neighborhoods = calc_knn_neighborhoods(
            bin_coords,
            num_neighbors=int(np.ceil(fraction * len(bin_coords))),
        )
        # holding out each neutral trait in turn
        nll = sum(
            tgs_calc_heldout_nll(
                calibration_neighborhoods=calibration_neighborhoods,
                counts=counts,
                trait=trait,
                trait_name=trait_names[trait],
                neutral_traits=neutral_traits,
                max_rate_ratio=max_rate_ratio,
            )
            for trait in neutral_traits
        )
        logging.info(
            "- tgs: calibration neighborhoods of "
            f"{fraction:.1%} of bins give "
            f"held-out neutral negative log likelihood {nll:.1f}",
        )
        if nll < best_nll:
            best_nll, best_neighborhoods = nll, calibration_neighborhoods
            best_fraction = fraction

    neutral_gains = (
        counts.loc[list(neutral_traits), "num_successes"]
        .groupby(level="bin")
        .sum()
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
