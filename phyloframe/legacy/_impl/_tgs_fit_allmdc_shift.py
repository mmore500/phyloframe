import typing

import numpy as np
from scipy import optimize

from ._tgs_calc_allmdc_group_nll import tgs_calc_allmdc_group_nll


def tgs_fit_allmdc_shift(
    *,
    group_data: typing.Sequence[typing.Optional[typing.Tuple]],
    params: np.ndarray,
    loss_rate: float,
    max_shift: float,
    max_growth_exponent: float,
) -> float:
    """Implementation detail for `alifestd_tgs_allmdc_estimate_asexual`.

    Here, tgs stands for trait gain/selection.

    Fits one clade-size group's shift, in [0, `max_shift`], to every
    trait's partitions in that group (`group_data`, per trait; None to
    skip), given each trait's (log gain rate, selection) in `params` (nan
    to skip).
    """
    used = [
        (data, log_gain_rate, selection - loss_rate)
        for data, (log_gain_rate, selection) in zip(group_data, params)
        if data is not None and len(data[0]) and np.isfinite(log_gain_rate)
    ]

    def nll(shift: float) -> float:
        return sum(
            tgs_calc_allmdc_group_nll(
                log_gain_rate=log_gain_rate,
                rate=rate,
                shift=shift,
                group_data=data,
                max_growth_exponent=max_growth_exponent,
            )[0]
            for data, log_gain_rate, rate in used
        )

    return float(
        optimize.minimize_scalar(
            nll,
            bounds=(0.0, max_shift),
            method="bounded",
            options=dict(xatol=max_shift * 1e-6),
        ).x
    )
