import typing

import numpy as np
import pandas as pd

from ._TgsAllmdcBranches import TgsAllmdcBranches
from ._tgs_init_results import tgs_init_results


def tgs_init_allmdc_results(
    *,
    trait_names: typing.Sequence[str],
    branches: TgsAllmdcBranches,
) -> pd.DataFrame:
    """Implementation detail for `alifestd_tgs_allmdc_estimate_asexual`.

    Here, tgs stands for trait gain/selection.

    Returns one row per trait, with counts of leaves and pendant gains, of
    inner branches (clades) and inner gains, and estimates for a trait with
    no fit (no gain, undefined selection).
    """
    is_leaf = branches.clade_sizes == 1
    res = tgs_init_results(
        trait_names=trait_names,
        trait_leaves=[units[is_leaf[units]] for units in branches.trait_units],
        trait_carriers=[
            gains[is_leaf[gains]] for gains in branches.trait_gains
        ],
    )
    res.insert(
        3,
        "num_clades",
        np.array(
            [np.count_nonzero(~is_leaf[u]) for u in branches.trait_units],
            dtype=np.int64,
        ),
    )
    res.insert(
        4,
        "num_clade_gains",
        np.array(
            [np.count_nonzero(~is_leaf[g]) for g in branches.trait_gains],
            dtype=np.int64,
        ),
    )
    return res
