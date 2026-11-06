import typing

import numpy as np
import polars as pl
import polars.selectors as cs


def tgs_extract_trait_masks(
    exposure_df: pl.DataFrame,
) -> typing.Tuple[np.ndarray, np.ndarray]:
    """Implementation detail for `alifestd_tgs_pendantmdc_estimate_polars` and
    `alifestd_tgs_allmdc_estimate_polars`. Acronym tgs stands for trait
    gain/selection.

    Returns boolean arrays of shape (num_branches, num_traits) marking,
    for each branch of `exposure_df` (see `tgs_calc_exposure_df`) and
    trait, whether the branch started without the trait, and whether it
    gained it.
    """
    starts_absent_df = exposure_df.select(
        cs.matches(r"^trait\d+_starts_absent$")
    )
    gains_df = exposure_df.select(cs.matches(r"^trait\d+_gain$"))
    return tuple(
        df.to_numpy().astype(bool).reshape(len(exposure_df), df.width)
        for df in (starts_absent_df, gains_df)
    )
