import typing

import numpy as np
import pandas as pd


def tgs_find_trait_exposures(
    exposure_df: pd.DataFrame,
) -> typing.Tuple[typing.List[np.ndarray], typing.List[np.ndarray]]:
    """Implementation detail for
    `alifestd_tgs_pendantmdc_estimate_asexual` and
    `alifestd_tgs_allmdc_estimate_asexual`.

    Here, tgs stands for trait gain/selection.

    Returns, per trait, positions in `exposure_df` (see `tgs_calc_exposure_df`)
    of branches that started without it (their parent lacks it), and of
    those that gained it.
    """
    num_traits = sum(
        column.endswith("_starts_absent") for column in exposure_df
    )
    return (
        [
            np.flatnonzero(exposure_df[f"trait{i}_starts_absent"].to_numpy())
            for i in range(num_traits)
        ],
        [
            np.flatnonzero(exposure_df[f"trait{i}_gain"].to_numpy())
            for i in range(num_traits)
        ],
    )
