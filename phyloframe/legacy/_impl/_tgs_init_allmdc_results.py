import typing

import pandas as pd

from ._tgs_find_trait_branches import tgs_find_trait_branches
from ._tgs_init_results import tgs_init_results


def tgs_init_allmdc_results(
    *,
    trait_names: typing.Sequence[str],
    branches: pd.DataFrame,
) -> pd.DataFrame:
    """Implementation detail for `alifestd_tgs_allmdc_estimate_asexual`.

    Here, tgs stands for trait gain/selection.

    Returns one row per trait, with counts of leaf branches at risk of
    gaining it and of pendant gains, of inner branches (clades) at risk and
    of inner gains, from `branches` (see `tgs_calc_branches`), and
    estimates for a trait with no fit (no gain, undefined selection).
    """
    is_leaf = branches["clade_size"].to_numpy() == 1
    trait_units, trait_gains = tgs_find_trait_branches(branches)
    res = tgs_init_results(
        trait_names=trait_names,
        trait_leaves=[units[is_leaf[units]] for units in trait_units],
        trait_carriers=[gains[is_leaf[gains]] for gains in trait_gains],
    )
    res.insert(
        3, "num_clades", [int((~is_leaf[u]).sum()) for u in trait_units]
    )
    res.insert(
        4, "num_clade_gains", [int((~is_leaf[g]).sum()) for g in trait_gains]
    )
    return res
