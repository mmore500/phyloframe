import typing

import numpy as np
import opytional as opyt
import pandas as pd

from ._TgsAllmdcGroup import TgsAllmdcGroup
from ._tgs_calc_allmdc_group_data import tgs_calc_allmdc_group_data
from ._tgs_calc_gain_ratios import tgs_calc_gain_ratios
from ._tgs_fit_allmdc_shift import tgs_fit_allmdc_shift
from ._tgs_fit_allmdc_trait import tgs_fit_allmdc_trait


def tgs_fit_allmdc_traits(
    *,
    groups: typing.Sequence[TgsAllmdcGroup],
    unit_weights: typing.Sequence[np.ndarray],
    neutral_traits: typing.Sequence[int],
    calibrate: bool,
    loss_rate: float,
    max_shift: float,
    num_traits: int,
    starts: typing.Optional[np.ndarray],
    shifts_start: typing.Optional[np.ndarray],
    selection_starts: typing.Sequence[float],
    max_growth_exponent: float,
    max_shift_iterations: int,
    progress_wrap: typing.Callable,
) -> typing.Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Implementation detail for `alifestd_tgs_allmdc_estimate_asexual`.

    Here, tgs stands for trait gain/selection.

    Jointly fits every trait's gain rate and selection to branches of all
    clade-size groups, with one survival shift per group shared across
    traits; see `tgs_calc_allmdc_group_nll` for the model. Alternates, up
    to `max_shift_iterations` times (wrapped in `progress_wrap`), between
    fitting each trait given the shifts and each shift given the traits.
    `starts`, per-trait (log gain rate, selection), and `shifts_start`
    seed the fit, e.g., from a previous fit; None for none.

    Returns one row per trait, with columns `gain_rate`, `selection`,
    `selection_lrt_naive`, and `gain_ratio`; each group's fitted shift (0
    for the first, single-leaf group; nan for groups without gains); and
    per-trait (log gain rate, selection).
    """
    data = tgs_calc_allmdc_group_data(
        groups=groups,
        unit_weights=unit_weights,
        neutral_traits=neutral_traits,
        calibrate=calibrate,
        num_traits=num_traits,
    )
    has_gains = np.array(
        [
            any(d is not None and d[1].sum() > 0 for d in trait_data)
            for trait_data in data
        ],
        dtype=bool,
    )
    group_has_gains = [
        any(
            data[trait][j] is not None and data[trait][j][1].sum() > 0
            for trait in range(num_traits)
        )
        for j in range(len(groups))
    ]
    shifts = np.where(group_has_gains, 0.0, np.nan)
    shifts[0] = 0.0
    if shifts_start is not None:
        refit = np.isfinite(shifts) & np.isfinite(shifts_start)
        shifts[refit] = shifts_start[refit]
    params = opyt.apply_if_or_else(
        starts, np.array, lambda: np.full((num_traits, 2), np.nan)
    )
    lrts = np.full(num_traits, np.nan)

    previous_total = np.inf
    for __ in progress_wrap(range(max_shift_iterations)):
        total = 0.0
        for trait in np.flatnonzero(has_gains):
            log_gain_rate, selection, nll, lrts[trait] = tgs_fit_allmdc_trait(
                trait_data=data[trait],
                shifts=shifts,
                loss_rate=loss_rate,
                start=(
                    tuple(params[trait])
                    if np.isfinite(params[trait]).all()
                    else None
                ),
                selection_starts=selection_starts,
                max_growth_exponent=max_growth_exponent,
            )
            params[trait] = (log_gain_rate, selection)
            total += nll
        previous_shifts = shifts.copy()
        for j in np.flatnonzero(np.isfinite(shifts))[1:]:
            shifts[j] = tgs_fit_allmdc_shift(
                group_data=[
                    data[trait][j] if has_gains[trait] else None
                    for trait in range(num_traits)
                ],
                params=params,
                loss_rate=loss_rate,
                max_shift=max_shift,
                max_growth_exponent=max_growth_exponent,
            )
        is_shift_converged = (
            np.nanmax(np.abs(shifts - previous_shifts), initial=0.0)
            <= max_shift * 1e-4
        )
        is_nll_converged = abs(previous_total - total) <= 1e-9 * abs(total)
        previous_total = total
        if is_shift_converged and is_nll_converged:
            break

    gain_rates = np.where(has_gains, np.exp(params[:, 0]), 0.0)
    res = pd.DataFrame(
        {
            "gain_rate": gain_rates,
            "selection": np.where(has_gains, params[:, 1], np.nan),
            "selection_lrt_naive": np.where(has_gains, lrts, np.nan),
            "gain_ratio": tgs_calc_gain_ratios(
                gain_rates=gain_rates, neutral_traits=neutral_traits
            ),
        },
    )
    return res, shifts, params
