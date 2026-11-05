import typing

import numpy as np
import pandas as pd

from ._tgs_summarize_bootstrap import tgs_summarize_bootstrap


def tgs_build_estimates_df(
    counts_df: pd.DataFrame,
    *,
    point_df: typing.Optional[pd.DataFrame],
    replicate_fits_df: typing.Optional[pd.DataFrame],
    num_bootstrap: int,
) -> pd.DataFrame:
    """Implementation detail for `alifestd_tgs_pendantmdc_estimate_polars` and
    `alifestd_tgs_allmdc_estimate_polars`. Acronym tgs stands for trait
    gain/selection.

    Returns one row per trait, with `counts_df` columns (`trait`, then
    per-trait counts), then point estimates from `point_df`, then, with
    bootstrap replicates, summary columns (see `tgs_summarize_bootstrap`).
    Without `point_df`, i.e., with nothing to fit, each trait has no gain
    and undefined selection; without `replicate_fits_df`, every replicate
    matches the point fit.
    """
    columns = [
        "gain_rate",
        "selection",
        "gain_ratio",
        "selection_lrt_naive",
        "gain_ratio_lrt_naive",
    ]
    if point_df is None:
        point_df = pd.DataFrame(
            {"gain_rate": 0.0, **dict.fromkeys(columns[1:], np.nan)},
            index=counts_df.index,
        )
    estimates_df = pd.concat([counts_df, point_df[columns]], axis=1)
    if not num_bootstrap:
        return estimates_df
    if replicate_fits_df is None:
        replicate_fits_df = pd.concat(
            [estimates_df] * num_bootstrap,
            keys=range(num_bootstrap),
            names=["replicate", "trait_index"],
        )
    return pd.concat(
        [
            estimates_df,
            tgs_summarize_bootstrap(
                point_df=estimates_df, replicate_fits_df=replicate_fits_df
            ),
        ],
        axis=1,
    )
