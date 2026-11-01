import numpy as np
import pandas as pd

from ..._auxlib._sample_bootstrap_counts import sample_bootstrap_counts
from ._tgs_fit_allmdc_traits import tgs_fit_allmdc_traits


def tgs_fit_allmdc_bootstrap_replicate(
    *, seed: int, fit_kwargs: dict
) -> pd.DataFrame:
    """Implementation detail for `alifestd_tgs_allmdc_estimate_asexual`.

    Here, tgs stands for trait gain/selection.

    Returns fit to one bootstrap resample of branches; see
    `tgs_fit_allmdc_traits`, which `fit_kwargs` are passed to.
    """
    unit_weights = sample_bootstrap_counts(
        len(fit_kwargs["branches"]),
        rng=np.random.default_rng(seed),
    )
    return tgs_fit_allmdc_traits(
        unit_weights=unit_weights.astype(np.float64), **fit_kwargs
    )
