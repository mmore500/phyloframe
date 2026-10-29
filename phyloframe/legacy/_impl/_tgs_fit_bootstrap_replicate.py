import numpy as np
import pandas as pd

from ..._auxlib._sample_bootstrap_counts import sample_bootstrap_counts
from ._tgs_fit_traits import tgs_fit_traits


def tgs_fit_bootstrap_replicate(
    *, seed: int, fit_kwargs: dict
) -> pd.DataFrame:
    """Implementation detail for `tgs_bootstrap`.

    Here, tgs stands for trait gain/selection.

    Returns fit to one bootstrap resample of leaves; see `tgs_fit_traits`.
    """
    leaf_weights = sample_bootstrap_counts(
        len(fit_kwargs["leaf_partitions"]), rng=np.random.default_rng(seed)
    )
    fits, __ = tgs_fit_traits(
        leaf_weights=leaf_weights.astype(np.float64), **fit_kwargs
    )
    return fits
