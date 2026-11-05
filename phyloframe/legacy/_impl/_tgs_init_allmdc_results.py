import typing

import numpy as np
import pandas as pd

from ._tgs_calc_trait_masks import tgs_calc_trait_masks
from ._tgs_init_results import tgs_init_results


def tgs_init_allmdc_results(
    *,
    trait_names: typing.Sequence[str],
    exposure_df: pd.DataFrame,
) -> pd.DataFrame:
    """Implementation detail for `alifestd_tgs_allmdc_estimate_asexual`.

    Here, tgs stands for trait gain/selection.

    Returns one row per trait (see `tgs_init_results`), with counts, from
    `exposure_df` (see `tgs_calc_exposure_df`), of leaf branches starting
    without the trait, `num_leaves`, and of their gains,
    `num_trait_present`, and the same for inner branches, `num_clades` and
    `num_clade_gains`.
    """
    starts_absent, gains = tgs_calc_trait_masks(exposure_df)
    is_leaf = np.expand_dims((exposure_df["clade_size"] == 1).to_numpy(), 1)
    return tgs_init_results(
        trait_names=trait_names,
        counts={
            "num_leaves": (starts_absent & is_leaf).sum(axis=0),
            "num_trait_present": (gains & is_leaf).sum(axis=0),
            "num_clades": (starts_absent & ~is_leaf).sum(axis=0),
            "num_clade_gains": (gains & ~is_leaf).sum(axis=0),
        },
    )
