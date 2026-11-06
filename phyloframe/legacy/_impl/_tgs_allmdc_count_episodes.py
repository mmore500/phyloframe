import typing

import numpy as np
import polars as pl

from ._tgs_extract_trait_masks import tgs_extract_trait_masks


def tgs_allmdc_count_episodes(
    *,
    trait_names: typing.Sequence[str],
    exposure_df: pl.DataFrame,
) -> pl.DataFrame:
    """Implementation detail for `alifestd_tgs_allmdc_estimate_polars`. Acronym
    tgs stands for trait gain/selection.

    Returns one row per trait, with column `trait` and counts, from
    `exposure_df` (see `tgs_calc_exposure_df`), of leaf exposure episodes,
    i.e., leaf branches starting without the trait, `num_leaves`, and of
    their gains, `num_trait_present`, and the same for inner exposure
    episodes, `num_clades` and `num_clade_gains`.
    """
    starts_absent, gains = tgs_extract_trait_masks(exposure_df)
    is_leaf = np.expand_dims((exposure_df["clade_size"] == 1).to_numpy(), 1)
    return pl.DataFrame(
        {
            "trait": list(trait_names),
            "num_leaves": (starts_absent & is_leaf).sum(axis=0),
            "num_trait_present": (gains & is_leaf).sum(axis=0),
            "num_clades": (starts_absent & ~is_leaf).sum(axis=0),
            "num_clade_gains": (gains & ~is_leaf).sum(axis=0),
        },
    )
