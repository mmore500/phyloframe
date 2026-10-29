import typing

import numpy as np
import opytional as opyt
import pandas as pd
from scipy import sparse

from ..._auxlib._tgs_fit import TgsFitResult, _tgs_fit_impl
from ._tgs_calc_reference_rate_ratios import tgs_calc_reference_rate_ratios
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
    `alifestd_estimate_trait_gain_selection_pendant_asexual`.

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
    # leave each trait out of its own neutral reference
    refs = [
        sorted(set(neutral_traits) - {trait})
        for trait in range(len(trait_leaves))
    ]

    fits, calibration_factors = [], []
    for trait, (n, k, pendant_length_sum) in enumerate(zip(*counts)):
        pendant_length = np.divide(
            pendant_length_sum, n, out=np.zeros(n.shape), where=n > 0
        )
        factors = opyt.apply_if(
            calibration_neighborhoods,
            lambda neighborhoods: tgs_calc_reference_rate_ratios(
                calibration_neighborhoods=neighborhoods,
                counts=counts,
                refs=refs[trait],
            ),
        )
        if factors is not None:
            pendant_length *= factors
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
    ref_gain_rates = [
        res["gain_rate"][refs[trait]].mean() if len(refs[trait]) > 0 else 0.0
        for trait in range(len(res))
    ]
    res["gain_ratio"] = [
        gain_rate / ref_gain_rate if ref_gain_rate > 0 else np.nan
        for gain_rate, ref_gain_rate in zip(res["gain_rate"], ref_gain_rates)
    ]
    return res, calibration_factors
