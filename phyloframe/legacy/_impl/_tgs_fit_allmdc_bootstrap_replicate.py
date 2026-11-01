import typing

import numpy as np
import pandas as pd

from ..._auxlib._sample_bootstrap_counts import sample_bootstrap_counts
from ._tgs_fit_allmdc_traits import tgs_fit_allmdc_traits


def tgs_fit_allmdc_bootstrap_replicate(
    *, seed: int, fit_kwargs: dict
) -> typing.Tuple[pd.DataFrame, np.ndarray]:
    """Implementation detail for
    `alifestd_estimate_trait_gain_selection_allmdc_asexual`.

    Here, tgs stands for trait gain/selection.

    Returns fit and clade-size shifts for one bootstrap resample of
    branches, drawn with replacement across all clade-size groups; see
    `tgs_fit_allmdc_traits`.
    """
    groups = fit_kwargs["groups"]
    sizes = [len(group.partitions) for group in groups]
    weights = sample_bootstrap_counts(
        sum(sizes), rng=np.random.default_rng(seed)
    ).astype(np.float64)
    unit_weights = np.split(weights, np.cumsum(sizes)[:-1])
    fits, shifts, __ = tgs_fit_allmdc_traits(
        unit_weights=unit_weights, **fit_kwargs
    )
    return fits, shifts
