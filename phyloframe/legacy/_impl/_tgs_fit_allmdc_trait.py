import typing
import warnings

import numpy as np
from scipy import optimize

from ._tgs_calc_allmdc_group_nll import tgs_calc_allmdc_group_nll


def tgs_fit_allmdc_trait(
    *,
    trait_data: typing.Sequence[typing.Optional[typing.Tuple]],
    shifts: np.ndarray,
    loss_rate: float,
    start: typing.Optional[typing.Tuple[float, float]],
    selection_starts: typing.Sequence[float],
    max_growth_exponent: float,
) -> typing.Tuple[float, float, float, float]:
    """Implementation detail for `alifestd_tgs_allmdc_estimate_asexual`.

    Here, tgs stands for trait gain/selection.

    Fits one trait's gain rate and selection to its clade-size groups'
    partitions (see `tgs_calc_allmdc_group_data`), given clade-size
    `shifts`, ignoring groups that are None or have a nan shift. Starts
    from `start`, a previous (log gain rate, selection), if given, and from
    each of `selection_starts` (in units of 1 / mean length), as in
    `tgs_fit`. Returns log gain rate, selection, negative log likelihood,
    and the likelihood-ratio statistic for selection = 0.
    """
    used = [
        (data, shift)
        for data, shift in zip(trait_data, shifts)
        if data is not None and np.isfinite(shift) and len(data[0])
    ]
    num_trials = sum(data[0].sum() for data, __ in used)
    num_successes = sum(data[1].sum() for data, __ in used)
    exposure = sum((data[0] * data[2]).sum() for data, __ in used)
    # optimize selection in units of 1 / mean length, as in `tgs_fit`
    unit = num_trials / exposure

    def nll(params: np.ndarray) -> typing.Tuple[float, np.ndarray]:
        log_gain_rate, scaled_selection = params
        rate = scaled_selection * unit - loss_rate
        total, dlog_gain, drate = 0.0, 0.0, 0.0
        for data, shift in used:
            res, dlog, dr = tgs_calc_allmdc_group_nll(
                log_gain_rate=log_gain_rate,
                rate=rate,
                shift=shift,
                group_data=data,
                max_growth_exponent=max_growth_exponent,
            )
            total, dlog_gain, drate = total + res, dlog_gain + dlog, drate + dr
        return total, np.array([dlog_gain, drate * unit])

    def null_nll(params: np.ndarray) -> typing.Tuple[float, np.ndarray]:
        # selection = 0, i.e., rate = -loss_rate
        total, grad = nll(np.array([params[0], 0.0]))
        return total, grad[:1]

    minimize_kwargs = dict(jac=True, method="L-BFGS-B")
    starts = [
        [np.log(num_successes / exposure), selection_start]
        for selection_start in selection_starts
    ]
    if start is not None:
        starts.insert(0, [start[0], start[1] / unit])
    fits = [optimize.minimize(nll, x0, **minimize_kwargs) for x0 in starts]
    converged = [fit for fit in fits if fit.success]
    best = min(converged or fits, key=lambda res: res.fun)
    null = optimize.minimize(null_nll, best.x[:1], **minimize_kwargs)
    if not converged:
        warnings.warn(
            "tgs allmdc fit optimizer did not converge from any starting "
            f"point ({best.message}); estimates may be unreliable.",
        )
    return (
        float(best.x[0]),
        float(best.x[1] * unit),
        float(best.fun),
        float(2 * (null.fun - best.fun)),
    )
