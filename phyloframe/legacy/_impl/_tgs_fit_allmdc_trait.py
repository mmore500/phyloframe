import typing
import warnings

import numpy as np
from scipy import optimize

from ..._auxlib._tgs_fit import TgsFitResult
from ._tgs_calc_allmdc_data import TgsAllmdcData
from ._tgs_calc_allmdc_nll import tgs_calc_allmdc_nll


def tgs_fit_allmdc_trait(
    *,
    data: TgsAllmdcData,
    survival_bin_times: np.ndarray,
    loss_rate: float,
    selection_starts: typing.Sequence[float],
    max_growth_exponent: float,
) -> TgsFitResult:
    """Implementation detail for `alifestd_tgs_allmdc_estimate_asexual`.

    Here, tgs stands for trait gain/selection.

    Fits one trait's gain rate and selection to its partitions (see
    `tgs_calc_allmdc_data` and, for the model, `tgs_calc_allmdc_nll`),
    starting from each of `selection_starts` (in units of 1 / mean branch
    length) and keeping the best fit, as in `tgs_fit`.
    """
    num_trials, num_successes, lengths, __ = data
    exposure = np.sum(num_trials * lengths)
    # optimize selection in units of 1 / mean length, as in `tgs_fit`
    unit = num_trials.sum() / exposure

    def nll(params: np.ndarray) -> typing.Tuple[float, np.ndarray]:
        log_gain_rate, scaled_selection = params
        res, dlog_gain, drate = tgs_calc_allmdc_nll(
            log_gain_rate=log_gain_rate,
            rate=scaled_selection * unit - loss_rate,
            data=data,
            survival_bin_times=survival_bin_times,
            max_growth_exponent=max_growth_exponent,
        )
        return res, np.array([dlog_gain, drate * unit])

    def null_nll(params: np.ndarray) -> typing.Tuple[float, np.ndarray]:
        # selection = 0, i.e., rate = -loss_rate
        res, grad = nll(np.array([params[0], 0.0]))
        return res, grad[:1]

    minimize_kwargs = dict(jac=True, method="L-BFGS-B")
    log_gain_start = np.log(num_successes.sum() / exposure)
    fits = [
        optimize.minimize(
            nll, [log_gain_start, selection_start], **minimize_kwargs
        )
        for selection_start in selection_starts
    ]
    converged = [fit for fit in fits if fit.success]
    best = min(converged or fits, key=lambda res: res.fun)
    null = optimize.minimize(null_nll, best.x[:1], **minimize_kwargs)
    if not converged:
        warnings.warn(
            "tgs allmdc fit optimizer did not converge from any starting "
            f"point ({best.message}); estimates may be unreliable.",
        )
    if not null.success:
        warnings.warn(
            "tgs allmdc fit optimizer did not converge under the null "
            f"hypothesis of no selection ({null.message}); likelihood-ratio "
            "statistic may be unreliable.",
        )
    return TgsFitResult(
        gain_rate=float(np.exp(best.x[0])),
        selection=float(best.x[1] * unit),
        selection_lrt_naive=float(2 * (null.fun - best.fun)),
    )
