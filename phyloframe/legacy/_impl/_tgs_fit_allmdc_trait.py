import typing

import numpy as np
import pandas as pd
from scipy import sparse

from ._tgs_calc_allmdc_nll import tgs_calc_allmdc_nll
from ._tgs_fit import TgsFitResult, _tgs_minimize_nll


def tgs_fit_allmdc_trait(
    *,
    partitions: pd.DataFrame,
    survival_shares: sparse.csr_matrix,
    survival_bin_times: np.ndarray,
    loss_rate: float,
    selection_starts: typing.Sequence[float],
    growth_exponent_bound: float,
) -> typing.Dict[str, float]:
    """Implementation detail for `alifestd_tgs_allmdc_estimate_asexual`.

    Here, tgs stands for trait gain/selection.

    Fits one trait's gain rate and selection to its `partitions` and their
    `survival_shares` (see `tgs_calc_allmdc_data` and, for the model, `tgs_calc_allmdc_nll`),
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
    lengths = partitions["length"].to_numpy()
    exposure = np.sum(num_trials * lengths)
    return _tgs_minimize_nll(
        calc_nll=lambda log_gain_rate, rate: tgs_calc_allmdc_nll(
            log_gain_rate=log_gain_rate,
            rate=rate,
            partitions=partitions,
            survival_shares=survival_shares,
            survival_bin_times=survival_bin_times,
            growth_exponent_bound=growth_exponent_bound,
        ),
        # 1 / mean branch length, as in `tgs_fit`
        selection_scale=num_trials.sum() / exposure,
        log_gain_start=np.log(num_successes.sum() / exposure),
        loss_rate=loss_rate,
        selection_starts=selection_starts,
    )._asdict()
