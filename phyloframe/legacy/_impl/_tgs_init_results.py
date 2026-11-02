import typing

import numpy as np
import pandas as pd


def tgs_init_results(
    *,
    trait_names: typing.Sequence[str],
    counts: typing.Mapping[str, np.ndarray],
) -> pd.DataFrame:
    """Implementation detail for
    `alifestd_tgs_pendantmdc_estimate_asexual` and
    `alifestd_tgs_allmdc_estimate_asexual`.

    Here, tgs stands for trait gain/selection.

    Returns one row per trait, with columns `trait`, then per-trait
    `counts` in order, then estimates for a trait with no fit (no gain,
    undefined selection).
    """
    return pd.DataFrame(
        {
            "trait": list(trait_names),
            **counts,
            "gain_rate": 0.0,
            "selection": np.nan,
            "gain_ratio": np.nan,
            "selection_lrt_naive": np.nan,
            "gain_ratio_lrt_naive": np.nan,
        },
    )
