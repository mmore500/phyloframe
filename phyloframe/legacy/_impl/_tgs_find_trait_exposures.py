import typing

import numpy as np
import pandas as pd

from ._tgs_calc_trait_masks import tgs_calc_trait_masks


def tgs_find_trait_exposures(
    exposure_df: pd.DataFrame,
) -> typing.Tuple[typing.List[np.ndarray], typing.List[np.ndarray]]:
    """Implementation detail for `alifestd_tgs_pendantmdc_estimate_asexual` and
    `alifestd_tgs_allmdc_estimate_asexual`. Acronym tgs stands for trait
    gain/selection.

    Returns, per trait, positions in `exposure_df` (see `tgs_calc_exposure_df`)
    of branches that started without it (their parent lacks it), and of
    those that gained it.
    """
    starts_absent, gains = tgs_calc_trait_masks(exposure_df)
    return (
        [np.flatnonzero(mask) for mask in starts_absent.T],
        [np.flatnonzero(mask) for mask in gains.T],
    )
