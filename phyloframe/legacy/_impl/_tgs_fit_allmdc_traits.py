import typing
import warnings

import numpy as np
import pandas as pd
from scipy import optimize, sparse, special

from ..._auxlib._calc_binomial_nll_proxy import calc_binomial_nll_proxy
from ..._auxlib._tgs_fit import _tgs_calc_effective_length_deriv
from ._tgs_calc_calibrated_pendant_lengths import (
    tgs_calc_calibrated_pendant_lengths,
)
from ._tgs_count_traits import tgs_count_traits
from ._tgs_find_reference_traits import tgs_find_reference_traits

_SELECTION_STARTS = (0.0, -0.3, -1.8)
_MAX_GROWTH_EXPONENT = 600.0


class TgsAllmdcGroup(typing.NamedTuple):
    """Implementation detail for
    `alifestd_estimate_trait_gain_selection_allmdc_asexual`.

    Here, tgs stands for trait gain/selection.

    Branches of one clade-size group, binned into partitions: each branch's
    partition and length, per trait the positions (within the group) of
    branches that started without the trait and of those that gained it,
    and partition-by-partition calibration neighborhoods (None without
    calibration).
    """

    partitions: np.ndarray
    num_partitions: int
    branch_lengths: np.ndarray
    trait_units: typing.List[np.ndarray]
    trait_gains: typing.List[np.ndarray]
    calibration_neighborhoods: typing.Optional[sparse.spmatrix]


def _calc_group_data(
    *,
    groups: typing.Sequence[TgsAllmdcGroup],
    unit_weights: typing.Sequence[np.ndarray],
    neutral_traits: typing.Sequence[int],
    calibrate: bool,
    num_traits: int,
) -> typing.List[typing.List[typing.Optional[typing.Tuple]]]:
    """Per trait, per group: partitions' (weighted) branch counts, gain
    counts, and mean lengths, calibrated against neutral traits and put on
    the single-leaf group's neutral clock if `calibrate`; None where a
    group cannot be used for that trait."""
    counts = [
        tgs_count_traits(
            leaf_partitions=group.partitions,
            num_partitions=group.num_partitions,
            pendant_lengths=group.branch_lengths,
            leaf_weights=weights,
            trait_leaves=group.trait_units,
            trait_carriers=group.trait_gains,
        )
        for group, weights in zip(groups, unit_weights)
    ]
    data = [[None] * len(groups) for __ in range(num_traits)]
    for trait in range(num_traits):
        refs = tgs_find_reference_traits(
            neutral_traits=neutral_traits, trait=trait
        )
        clock = None  # single-leaf group's neutral gain rate
        for j, (group, cnt) in enumerate(zip(groups, counts)):
            lengths, factors = tgs_calc_calibrated_pendant_lengths(
                counts=cnt,
                trait=trait,
                neutral_traits=neutral_traits,
                calibration_neighborhoods=(
                    group.calibration_neighborhoods if calibrate else None
                ),
            )
            if calibrate:
                # neutral reference gain rate per calibrated unit length;
                # rescaling each group's lengths to match the single-leaf
                # group's makes neutral traits gain at one rate everywhere,
                # whatever distorts inner versus pendant branch lengths
                exposure = cnt.pendant_length_sums[refs].sum(axis=0) * factors
                with np.errstate(divide="ignore", invalid="ignore"):
                    rate = cnt.num_carriers[refs].sum() / exposure.sum()
                if not np.isfinite(rate) or rate <= 0:
                    continue  # no neutral gains here to calibrate against
                if j == 0:
                    clock = rate
                if clock is None:
                    continue
                lengths = lengths * rate / clock
            n = cnt.num_leaves[trait]
            keep = n > 0
            data[trait][j] = (
                n[keep],
                cnt.num_carriers[trait][keep],
                lengths[keep],
            )
    return data


def _calc_group_nll(
    log_gain: float,
    rate: float,
    shift: float,
    group_data: typing.Tuple,
) -> typing.Tuple[float, float, float]:
    """Negative log likelihood of one group's partitions, and its
    derivatives with respect to log gain rate and to `rate`."""
    n, k, x = group_data
    gain = np.exp(log_gain)
    effective_length = x * special.exprel(
        np.minimum(rate * x, _MAX_GROWTH_EXPONENT)
    )
    effective_length_deriv = _tgs_calc_effective_length_deriv(
        pendant_length=x, rate=rate, max_growth_exponent=_MAX_GROWTH_EXPONENT
    )
    # survival over the shift, i.e., the time a clade of this size takes to
    # grow out to its sampled leaves
    shift_factor = np.exp(min(rate * shift, _MAX_GROWTH_EXPONENT))
    p = gain * effective_length * shift_factor
    nll, dnll_dp = calc_binomial_nll_proxy(n, k, p)
    dnll_dlog_gain = np.sum(dnll_dp * p)
    dnll_drate = np.sum(
        dnll_dp
        * gain
        * shift_factor
        * (effective_length_deriv + shift * effective_length)
    )
    return nll, dnll_dlog_gain, dnll_drate


def _fit_trait(
    trait_data: typing.Sequence[typing.Optional[typing.Tuple]],
    shifts: np.ndarray,
    loss_rate: float,
    start: typing.Optional[typing.Tuple[float, float]],
) -> typing.Tuple[float, float, float, float]:
    """Returns log gain rate, selection, negative log likelihood, and the
    likelihood-ratio statistic for selection = 0, given clade-size shifts."""
    used = [
        (data, shift)
        for data, shift in zip(trait_data, shifts)
        if data is not None and np.isfinite(shift) and len(data[0])
    ]
    n_total = sum(d[0].sum() for d, __ in used)
    k_total = sum(d[1].sum() for d, __ in used)
    nx_total = sum((d[0] * d[2]).sum() for d, __ in used)
    # optimize selection in units of 1 / mean length, as in `tgs_fit`
    unit = n_total / nx_total

    def nll(params: np.ndarray) -> typing.Tuple[float, np.ndarray]:
        log_gain, scaled_selection = params
        rate = scaled_selection * unit - loss_rate
        total, dlog_gain, drate = 0.0, 0.0, 0.0
        for data, shift in used:
            a, b, c = _calc_group_nll(log_gain, rate, shift, data)
            total, dlog_gain, drate = total + a, dlog_gain + b, drate + c
        return total, np.array([dlog_gain, drate * unit])

    def null_nll(params: np.ndarray) -> typing.Tuple[float, np.ndarray]:
        # at selection = 0, i.e., scaled selection 0, rate = -loss_rate
        total, grad = nll(np.array([params[0], 0.0]))
        return total, grad[:1]

    minimize_kwargs = dict(jac=True, method="L-BFGS-B")
    starts = [[np.log(k_total / nx_total), s] for s in _SELECTION_STARTS]
    if start is not None:  # previous fit, as (log gain rate, selection)
        starts.insert(0, [start[0], start[1] / unit])
    fits = [optimize.minimize(nll, x0, **minimize_kwargs) for x0 in starts]
    converged = [fit for fit in fits if fit.success]
    best = min(converged or fits, key=lambda res: res.fun)
    null = optimize.minimize(null_nll, best.x[:1], **minimize_kwargs)
    if not converged:
        warnings.warn(
            "tgs clade fit optimizer did not converge from any starting "
            f"point ({best.message}); estimates may be unreliable.",
        )
    return (
        float(best.x[0]),
        float(best.x[1] * unit),
        float(best.fun),
        float(2 * (null.fun - best.fun)),
    )


def tgs_fit_allmdc_traits(
    *,
    groups: typing.Sequence[TgsAllmdcGroup],
    unit_weights: typing.Sequence[np.ndarray],
    neutral_traits: typing.Sequence[int],
    calibrate: bool,
    loss_rate: float,
    max_shift: float,
    num_traits: int,
    starts: typing.Optional[np.ndarray] = None,
    shifts_start: typing.Optional[np.ndarray] = None,
    max_iterations: int = 50,
) -> typing.Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Implementation detail for
    `alifestd_estimate_trait_gain_selection_allmdc_asexual`.

    Here, tgs stands for trait gain/selection.

    Jointly fits every trait's gain rate and selection to branches of all
    clade-size groups, with one survival shift per clade-size group shared
    across traits: a branch in group `j` gains the trait (its whole clade
    carries it) with probability

        gain_rate * (exp(c * length) - 1) / c * exp(c * shift_j)

    where `shift_0` = 0 for single leaves. Alternates between fitting each
    trait given the shifts and each shift given the traits.

    Returns one row per trait, with columns `gain_rate`, `selection`,
    `selection_lrt_naive`, and `gain_ratio`; the fitted shifts (nan for
    groups without gains); and per-trait fitted parameters (log gain rate,
    selection), as starting points for bootstrap replicates.
    """
    data = _calc_group_data(
        groups=groups,
        unit_weights=unit_weights,
        neutral_traits=neutral_traits,
        calibrate=calibrate,
        num_traits=num_traits,
    )
    num_groups = len(groups)
    has_gains = [
        sum(d[1].sum() for d in trait_data if d is not None) > 0
        for trait_data in data
    ]
    group_gains = np.array(
        [
            sum(
                data[trait][j][1].sum()
                for trait in range(num_traits)
                if data[trait][j] is not None
            )
            for j in range(num_groups)
        ]
    )
    shifts = np.where(group_gains > 0, 0.0, np.nan)
    shifts[0] = 0.0
    if shifts_start is not None:
        fill = np.isfinite(shifts) & np.isfinite(shifts_start)
        shifts[fill] = shifts_start[fill]

    params = np.full((num_traits, 2), np.nan)
    if starts is not None:
        params[:] = starts
    lrts = np.full(num_traits, np.nan)
    previous_total = np.inf
    for __ in range(max_iterations):
        total = 0.0
        for trait in range(num_traits):
            if not has_gains[trait]:
                continue
            start = (
                tuple(params[trait])
                if np.isfinite(params[trait]).all()
                else None
            )
            log_gain, selection, nll, lrt = _fit_trait(
                data[trait], shifts, loss_rate, start
            )
            params[trait] = (log_gain, selection)
            lrts[trait] = lrt
            total += nll
        old_shifts = shifts.copy()
        for j in range(1, num_groups):
            if not np.isfinite(shifts[j]):
                continue

            def group_nll(shift: float, j: int = j) -> float:
                return sum(
                    _calc_group_nll(
                        params[trait, 0],
                        params[trait, 1] - loss_rate,
                        shift,
                        data[trait][j],
                    )[0]
                    for trait in range(num_traits)
                    if has_gains[trait]
                    and data[trait][j] is not None
                    and len(data[trait][j][0])
                )

            shifts[j] = optimize.minimize_scalar(
                group_nll,
                bounds=(0.0, max_shift),
                method="bounded",
                options=dict(xatol=max_shift * 1e-6),
            ).x
        converged = np.nanmax(
            np.abs(shifts - old_shifts), initial=0.0
        ) <= max_shift * 1e-4 and abs(previous_total - total) <= 1e-9 * abs(
            total
        )
        previous_total = total
        if converged:
            break

    gain_rates = np.where(has_gains, np.exp(params[:, 0]), 0.0)
    selections = np.where(has_gains, params[:, 1], np.nan)
    res = pd.DataFrame(
        {
            "gain_rate": gain_rates,
            "selection": selections,
            "selection_lrt_naive": np.where(has_gains, lrts, np.nan),
        },
    )
    # normalize each gain rate by the mean gain rate of its neutral
    # reference, as in `tgs_fit_traits`
    refs = [
        tgs_find_reference_traits(neutral_traits=neutral_traits, trait=trait)
        for trait in range(num_traits)
    ]
    ref_gain_rates = [
        gain_rates[ref].mean() if len(ref) else 0.0 for ref in refs
    ]
    res["gain_ratio"] = [
        gain_rate / ref if ref > 0 else np.nan
        for gain_rate, ref in zip(gain_rates, ref_gain_rates)
    ]
    return res, shifts, params
