import logging
import typing

import numpy as np
import pandas as pd

from ..._auxlib._calc_group_means import calc_group_means


def tgs_bin_survival_times(
    survival_times: np.ndarray, *, max_bins: int
) -> typing.Tuple[np.ndarray, np.ndarray]:
    """Implementation detail for `alifestd_tgs_allmdc_estimate_asexual`.

    Here, tgs stands for trait gain/selection.

    Returns each branch's survival-time bin, among up to `max_bins`
    equal-occupancy bins (merging those with tied bounds), and each bin's
    mean survival time.
    """
    # constant survival times give one bin, which pd.qcut leaves nan
    bins = (
        pd.Series(
            pd.qcut(survival_times, max_bins, labels=False, duplicates="drop")
        )
        .fillna(0)
        .to_numpy(dtype=np.int64)
    )
    num_bins = bins.max() + 1
    occupancy = np.bincount(bins, minlength=num_bins)
    logging.info(
        "- alifestd_tgs_allmdc_estimate_asexual: binned survival times into "
        f"{num_bins} bins, holding {occupancy.min()}-{occupancy.max()} "
        "branches each",
    )
    return bins, calc_group_means(bins, survival_times, num_groups=num_bins)
