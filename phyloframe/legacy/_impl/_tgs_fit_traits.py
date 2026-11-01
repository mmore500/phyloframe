import typing

import numpy as np
import opytional as opyt
import pandas as pd
from scipy import sparse

from ..._auxlib._tgs_fit import TgsFitResult, _tgs_fit_impl
from ._tgs_calc_calibrated_pendant_lengths import (
    tgs_calc_calibrated_pendant_lengths,
)
from ._tgs_calc_gain_ratios import tgs_calc_gain_ratios
from ._tgs_count_traits import tgs_count_traits


def tgs_fit_traits(
    *,
    leaf_partitions: np.ndarray,
    num_partitions: int,
    pendant_lengths: np.ndarray,
    leaf_weights: np.ndarray,
    trait_leaves: typing.Sequence[np.ndarray],
    trait_carriers: typing.Sequence[np.ndarray],
    neutral_traits: typing.Sequence[int],
    calibration_neighborhoods: typing.Optional[sparse.spmatrix],
    loss_rate: float,
) -> typing.Tuple[pd.DataFrame, typing.List[typing.Optional[np.ndarray]]]:
    """Implementation detail for
    `alifestd_tgs_pendantmdc_estimate_asexual`.

    Here, tgs stands for trait gain/selection.

    Returns one row per trait, with columns of `TgsFitResult` plus
    `gain_ratio`, the gain rate relative to neutral traits; and, per trait,
    the factors rescaling pendant length in partitions with its leaves
    (None without calibration). If `calibration_neighborhoods` is given,
    each partition's pendant length is rescaled by the rate ratio of neutral
    trait gains within its neighborhood.
    """
    counts = tgs_count_traits(
        leaf_partitions=leaf_partitions,
        num_partitions=num_partitions,
        pendant_lengths=pendant_lengths,
        leaf_weights=leaf_weights,
        trait_leaves=trait_leaves,
        trait_carriers=trait_carriers,
    )
    fits, calibration_factors = [], []
    for trait, (n, k, __) in enumerate(zip(*counts)):
        pendant_length, factors = tgs_calc_calibrated_pendant_lengths(
            counts=counts,
            trait=trait,
            neutral_traits=neutral_traits,
            calibration_neighborhoods=calibration_neighborhoods,
        )
        calibration_factors.append(opyt.apply_if(factors, lambda f: f[n > 0]))
        # without carriers, skip fitting (and its warning), as callers report
        # such traits and replicates
        fits.append(
            _tgs_fit_impl(
                num_trials=n,
                num_successes=k,
                pendant_length=pendant_length,
                loss_rate=loss_rate,
            )
            if k.any()
            else TgsFitResult(
                gain_rate=0.0, selection=np.nan, selection_lrt_naive=np.nan
            )
        )

    # normalize each gain rate by the mean gain rate of its neutral reference;
    # undefined without neutral reference traits, or if none gained the trait
    res = pd.DataFrame(fits, columns=TgsFitResult._fields)
    res["gain_ratio"] = tgs_calc_gain_ratios(
        gain_rates=res["gain_rate"].to_numpy(),
        neutral_traits=neutral_traits,
    )
    return res, calibration_factors
