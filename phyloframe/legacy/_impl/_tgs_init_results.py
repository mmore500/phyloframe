import typing

import numpy as np
import pandas as pd


def tgs_init_results(
    *,
    trait_names: typing.Sequence[str],
    trait_leaves: typing.Sequence[np.ndarray],
    trait_carriers: typing.Sequence[np.ndarray],
) -> pd.DataFrame:
    """Implementation detail for
    `alifestd_estimate_trait_gain_selection_pendant_asexual`.

    Here, tgs stands for trait gain/selection.

    Returns one row per trait, with leaf and carrier counts, and estimates
    for a trait with no fit (no gain, undefined selection).
    """
    return pd.DataFrame(
        (
            dict(
                trait=name,
                num_leaves=len(leaves),
                num_trait_present=len(carriers),
                gain_rate=0.0,
                selection=np.nan,
                gain_ratio=np.nan,
                selection_lrt_naive=np.nan,
            )
            for name, leaves, carriers in zip(
                trait_names, trait_leaves, trait_carriers
            )
        ),
        columns=[
            "trait",
            "num_leaves",
            "num_trait_present",
            "gain_rate",
            "selection",
            "gain_ratio",
            "selection_lrt_naive",
        ],
    )
