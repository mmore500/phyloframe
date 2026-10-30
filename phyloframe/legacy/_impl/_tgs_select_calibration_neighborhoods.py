import logging
import typing
import warnings

import numpy as np
from scipy import sparse

from ..._auxlib._calc_knn_neighborhoods import calc_knn_neighborhoods
from ._TgsCounts import TgsCounts
from ._tgs_calc_heldout_nll import tgs_calc_heldout_nll


def tgs_select_calibration_neighborhoods(
    *,
    partition_coords: np.ndarray,
    neighborhood_fractions: typing.Sequence[float],
    counts: TgsCounts,
    neutral_traits: typing.Sequence[int],
    trait_names: typing.Sequence[str],
) -> sparse.spmatrix:
    """Implementation detail for
    `alifestd_estimate_trait_gain_selection_pendant_asexual`.

    Here, tgs stands for trait gain/selection.

    Returns partition-by-partition neighborhood membership matrix for the
    neighborhood size, among `neighborhood_fractions`, that best predicts
    each held-out neutral trait's carriers when calibrating against the
    others, warning about any neighborhoods without neutral gains.
    """
    best_nll, best_neighborhoods = np.inf, None
    for fraction in neighborhood_fractions:
        # each partition's nearest partitions, itself included, together
        # holding `fraction` of leaves, as partitions are near-equal in size
        calibration_neighborhoods = calc_knn_neighborhoods(
            partition_coords,
            num_neighbors=int(np.ceil(fraction * len(partition_coords))),
        )
        nll = tgs_calc_heldout_nll(
            calibration_neighborhoods=calibration_neighborhoods,
            counts=counts,
            neutral_traits=neutral_traits,
            trait_names=trait_names,
        )
        logging.info(
            "- alifestd_estimate_trait_gain_selection_pendant_asexual: "
            f"calibration neighborhoods of {fraction:.1%} of leaves give "
            f"held-out neutral negative log likelihood {nll:.1f}",
        )
        if nll < best_nll:
            best_nll, best_neighborhoods = nll, calibration_neighborhoods

    neutral_gains = counts.num_carriers[neutral_traits].sum(axis=0)
    num_empty = int((best_neighborhoods @ neutral_gains == 0).sum())
    if num_empty:
        warnings.warn(
            f"{num_empty} of {len(partition_coords)} calibration "
            "neighborhoods contain no neutral trait gains, so leaves there "
            "get zero effective pendant length; more neutral traits or "
            "leaves would help.",
        )
    return best_neighborhoods
