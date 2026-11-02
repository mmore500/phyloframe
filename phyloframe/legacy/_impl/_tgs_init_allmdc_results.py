import typing

import pandas as pd

from ._tgs_find_trait_exposures import tgs_find_trait_exposures
from ._tgs_init_results import tgs_init_results


def tgs_init_allmdc_results(
    *,
    trait_names: typing.Sequence[str],
    exposure_df: pd.DataFrame,
) -> pd.DataFrame:
    """Implementation detail for `alifestd_tgs_allmdc_estimate_asexual`.

    Here, tgs stands for trait gain/selection.

    Returns one row per trait, with counts of leaf branches starting without
    the trait and of pendant gains, and of inner branches (clades) starting
    without it and of inner gains, from `exposure_df` (see
    `tgs_calc_exposure_df`), and estimates for a trait with no fit (no gain,
    undefined selection).
    """
    is_leaf = exposure_df["clade_size"].to_numpy() == 1
    trait_units, trait_gains = tgs_find_trait_exposures(exposure_df)
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
