import typing

import numpy as np
import pandas as pd

from ._TgsAllmdcUnits import TgsAllmdcUnits
from ._tgs_init_results import tgs_init_results
from ._tgs_name_allmdc_shifts import tgs_name_allmdc_shifts


def tgs_init_allmdc_results(
    *,
    trait_names: typing.Sequence[str],
    units: TgsAllmdcUnits,
) -> pd.DataFrame:
    """Implementation detail for `alifestd_tgs_allmdc_estimate_asexual`.

    Here, tgs stands for trait gain/selection.

    Returns one row per trait, with leaf and pendant gain counts, inner
    branch and inner gain counts, estimates for a trait with no fit (no
    gain, undefined selection), and a nan shift per clade-size group after
    the first.
    """
    res = tgs_init_results(
        trait_names=trait_names,
        trait_leaves=units.trait_units[0],
        trait_carriers=units.trait_gains[0],
    )
    res.insert(
        3,
        "num_clades",
        [sum(map(len, per_trait)) for per_trait in zip(*units.trait_units[1:])]
        or 0,
    )
    res.insert(
        4,
        "num_clade_gains",
        [sum(map(len, per_trait)) for per_trait in zip(*units.trait_gains[1:])]
        or 0,
    )
    for column in tgs_name_allmdc_shifts(units.group_bounds):
        res[column] = np.nan
    return res
