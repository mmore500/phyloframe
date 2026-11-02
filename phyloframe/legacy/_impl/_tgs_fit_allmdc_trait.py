import typing

import numpy as np
import pandas as pd
from scipy import sparse

from ._tgs_calc_allmdc_nll import tgs_calc_allmdc_nll
from ._tgs_fit import TgsFitResult, _tgs_minimize_nll


def tgs_make_allmdc_trait_nll(
    *,
    partitions: pd.DataFrame,
    survival_shares: sparse.csr_matrix,
    survival_bin_times: np.ndarray,
    growth_exponent_bound: float,
) -> typing.Tuple[typing.Callable, float]:
    """Implementation detail for `alifestd_tgs_allmdc_estimate_asexual`.

    Here, tgs stands for trait gain/selection.

    Returns `calc_nll(log gain rate, rate)` for one trait's `partitions`
    and their `survival_shares` (see `tgs_calc_allmdc_nll`), and the
    selection scale, 1 / mean branch length, as in `tgs_fit`.
    """
    num_trials = partitions["num_trials"].to_numpy()
    lengths = partitions["length"].to_numpy()

    def calc_nll(
        log_gain_rate: float, rate: float
    ) -> typing.Tuple[float, float, float]:
        return tgs_calc_allmdc_nll(
            log_gain_rate=log_gain_rate,
            rate=rate,
            partitions=partitions,
            survival_shares=survival_shares,
            survival_bin_times=survival_bin_times,
            growth_exponent_bound=growth_exponent_bound,
        )

    return calc_nll, num_trials.sum() / np.sum(num_trials * lengths)


def tgs_fit_allmdc_trait(
    *,
    calc_nll: typing.Callable,
    selection_scale: float,
    partitions: pd.DataFrame,
    loss_rate: float,
    selection_starts: typing.Sequence[float],
) -> typing.Dict[str, float]:
    """Implementation detail for `alifestd_tgs_allmdc_estimate_asexual`.

    Here, tgs stands for trait gain/selection.

    Fits one trait's gain rate and selection to its `partitions` (see
    `tgs_calc_allmdc_data`), with `calc_nll` and `selection_scale` from
    `tgs_make_allmdc_trait_nll` (for the model, see `tgs_calc_allmdc_nll`),
    starting from each of `selection_starts` (in units of 1 / mean branch
    length) and keeping the best fit, as in `tgs_fit`. Returns a dict with
    the fields of `TgsFitResult`; without gains, skips fitting, giving
    zero gain rate and undefined selection.
    """
    num_trials = partitions["num_trials"].to_numpy()
    num_successes = partitions["num_successes"].to_numpy()
    if not num_successes.any():
        return TgsFitResult(
            gain_rate=0.0, selection=np.nan, selection_lrt_naive=np.nan
        )._asdict()
    exposure = np.sum(num_trials * partitions["length"].to_numpy())
    return _tgs_minimize_nll(
        calc_nll=calc_nll,
        selection_scale=selection_scale,
        log_gain_start=np.log(num_successes.sum() / exposure),
        loss_rate=loss_rate,
        selection_starts=selection_starts,
    )._asdict()
