import typing

import numpy as np
import pandas as pd
from scipy import sparse

from ._tgs_count_traits import tgs_count_traits
from ._tgs_fit_trait_partitions import tgs_fit_trait_partitions
from ._tgs_tabulate_partitions import tgs_tabulate_partitions


def tgs_fit_traits(
    *,
    leaf_partitions: np.ndarray,
    num_partitions: int,
    pendant_lengths: np.ndarray,
    weights: np.ndarray,
    trait_leaves: typing.Sequence[np.ndarray],
    trait_carriers: typing.Sequence[np.ndarray],
    neutral_traits: typing.Sequence[int],
    calibration_neighborhoods: typing.Optional[sparse.spmatrix],
    max_rate_ratio: float,
    loss_rate: float,
    selection_starts: typing.Sequence[float],
    growth_exponent_bound: float,
) -> pd.DataFrame:
    """Implementation detail for `alifestd_tgs_pendantmdc_estimate_polars`.
    Acronym tgs stands for trait gain/selection.

    Fits each trait to its leaves' pendant branches, weighted by `weights`
    (e.g., bootstrap draw counts), returning one row per trait (see
    `tgs_fit_trait_partitions`). If `calibration_neighborhoods` is given,
    each partition's pendant length is rescaled by the rate ratio of
    neutral trait gains within its neighborhood, capped at
    `max_rate_ratio`.
    """
    counts = tgs_count_traits(
        leaf_partitions=leaf_partitions,
        num_partitions=num_partitions,
        pendant_lengths=pendant_lengths,
        leaf_weights=weights,
        trait_leaves=trait_leaves,
        trait_carriers=trait_carriers,
    )
    data = tgs_tabulate_partitions(
        counts=counts,
        neutral_traits=neutral_traits,
        calibration_neighborhoods=calibration_neighborhoods,
        max_rate_ratio=max_rate_ratio,
    )
    return tgs_fit_trait_partitions(
        data=data,
        neutral_traits=neutral_traits,
        loss_rate=loss_rate,
        selection_starts=selection_starts,
        growth_exponent_bound=growth_exponent_bound,
    )
