import typing

import numpy as np
import pandas as pd


def tgs_extract_trait_masks(
    exposure_df: pd.DataFrame,
) -> typing.Tuple[np.ndarray, np.ndarray]:
    """Implementation detail for `alifestd_tgs_pendantmdc_estimate_polars` and
    `alifestd_tgs_allmdc_estimate_polars`. Acronym tgs stands for trait
    gain/selection.

    Returns boolean arrays of shape (num_branches, num_traits) marking,
    for each branch of `exposure_df` (see `tgs_calc_exposure_df`) and
    trait, whether the branch started without the trait, and whether it
    gained it.
    """
    num_traits = sum(
        column.endswith("_starts_absent") for column in exposure_df
    )
    traits = range(num_traits)
    return (
        exposure_df[[f"trait{i}_starts_absent" for i in traits]]
        .to_numpy(dtype=bool)
        .reshape(len(exposure_df), num_traits),
        exposure_df[[f"trait{i}_gain" for i in traits]]
        .to_numpy(dtype=bool)
        .reshape(len(exposure_df), num_traits),
    )
