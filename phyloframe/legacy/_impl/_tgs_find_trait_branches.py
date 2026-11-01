import typing

import numpy as np
import pandas as pd


def tgs_find_trait_branches(
    branches: pd.DataFrame,
) -> typing.Tuple[typing.List[np.ndarray], typing.List[np.ndarray]]:
    """Implementation detail for
    `alifestd_tgs_pendantmdc_estimate_asexual` and
    `alifestd_tgs_allmdc_estimate_asexual`.

    Here, tgs stands for trait gain/selection.

    Returns, per trait, positions in `branches` (see `tgs_calc_branches`)
    of branches at risk of gaining it, and of those that gained it.
    """
    num_traits = sum(column.endswith("_at_risk") for column in branches)
    return (
        [
            np.flatnonzero(branches[f"trait{i}_at_risk"].to_numpy())
            for i in range(num_traits)
        ],
        [
            np.flatnonzero(branches[f"trait{i}_gain"].to_numpy())
            for i in range(num_traits)
        ],
    )
