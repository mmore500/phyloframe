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
    partition_coords: np.ndarray,
    neighborhood_fractions: typing.Sequence[float],
    counts: pd.DataFrame,
    neutral_traits: typing.Sequence[int],
    trait_names: typing.Sequence[str],
    max_rate_ratio: float,
) -> sparse.spmatrix:
    """Implementation detail for `alifestd_tgs_pendantmdc_estimate_polars` and
    `alifestd_tgs_allmdc_estimate_polars`. Acronym tgs stands for trait
    gain/selection.

    Returns the partition-by-partition membership matrix of calibration
    neighborhoods: each partition's nearest partitions by
    `partition_coords`, of the size, among `neighborhood_fractions`, that
    best predicts each held-out neutral trait's gains when calibrating
    against the other neutral traits (see `tgs_calc_heldout_nll`), given
    (`trait`, `partition`)-indexed `counts` (see `tgs_count_traits`).
    Warns about any neighborhoods without neutral gains.
    """
    best_nll, best_neighborhoods, best_fraction = np.inf, None, np.nan
    for fraction in neighborhood_fractions:
        # each partition's nearest partitions, itself included, together
        # holding about `fraction` of partitions
        calibration_neighborhoods = calc_knn_neighborhoods(
            partition_coords,
            num_neighbors=int(np.ceil(fraction * len(partition_coords))),
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
            f"{fraction:.1%} of partitions give "
            f"held-out neutral negative log likelihood {nll:.1f}",
        )
        if nll < best_nll:
            best_nll, best_neighborhoods = nll, calibration_neighborhoods
            best_fraction = fraction

    neutral_gains = (
        counts.loc[list(neutral_traits), "num_successes"]
        .groupby(level="partition")
        .sum()
        .to_numpy()
    )
    num_empty = int((best_neighborhoods @ neutral_gains == 0).sum())
    logging.info(
        "- tgs: selected calibration "
        f"neighborhoods of {best_fraction:.1%} of {len(partition_coords)} "
        f"partitions, {num_empty} without neutral trait gains",
    )
    if num_empty:
        warnings.warn(
            f"{num_empty} of {len(partition_coords)} calibration "
            "neighborhoods contain no neutral trait gains, so branches "
            "there get zero effective length; more neutral traits or "
            "leaves would help.",
        )
    return best_neighborhoods
