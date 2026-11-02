import logging
import typing

import numpy as np
from scipy import sparse

from ._TgsCounts import TgsCounts
from ._tgs_calc_calibrated_pendant_lengths import (
    tgs_calc_calibrated_pendant_lengths,
)
from ._tgs_select_calibration_neighborhoods import (
    tgs_select_calibration_neighborhoods,
)


def tgs_select_pendant_calibration(
    *,
    counts: TgsCounts,
    partition_coords: np.ndarray,
    neighborhood_fractions: typing.Sequence[float],
    neutral_traits: typing.Sequence[int],
    trait_names: typing.Sequence[str],
) -> sparse.spmatrix:
    """Implementation detail for
    `alifestd_tgs_pendantmdc_estimate_asexual`.

    Here, tgs stands for trait gain/selection.

    Returns calibration neighborhoods selected from unweighted
    per-partition `counts` (see `tgs_select_calibration_neighborhoods`),
    logging the range of factors by which they rescale each trait's
    pendant lengths.
    """
    neighborhoods = tgs_select_calibration_neighborhoods(
        partition_coords=partition_coords,
        neighborhood_fractions=neighborhood_fractions,
        counts=counts,
        neutral_traits=neutral_traits,
        trait_names=trait_names,
    )
    for trait, name in enumerate(trait_names):
        __, factors = tgs_calc_calibrated_pendant_lengths(
            counts=counts,
            trait=trait,
            neutral_traits=neutral_traits,
            calibration_neighborhoods=neighborhoods,
        )
        factors = factors[counts.num_leaves[trait] > 0]
        if len(factors):
            logging.info(
                "- alifestd_tgs_pendantmdc_estimate_asexual: calibrated "
                f"trait {name}, rescaling pendant lengths by "
                f"{factors.min():.3g} to {factors.max():.3g} (median "
                f"{np.median(factors):.3g})",
            )
    return neighborhoods
