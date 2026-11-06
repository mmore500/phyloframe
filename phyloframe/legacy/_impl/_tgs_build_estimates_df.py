import typing

import pandas as pd

from ._tgs_summarize_bootstrap import tgs_summarize_bootstrap


def tgs_build_estimates_df(
    counts_df: pd.DataFrame,
    *,
    point_estimate_df: pd.DataFrame,
    bootstrap_fits_df: typing.Optional[pd.DataFrame],
    num_bootstrap: int,
) -> pd.DataFrame:
    """Implementation detail for `alifestd_tgs_pendantmdc_estimate_polars` and
    `alifestd_tgs_allmdc_estimate_polars`. Acronym tgs stands for trait
    gain/selection.

    Returns one row per trait, with `counts_df` columns (`trait`, then
    per-trait counts), then point estimates from `point_estimate_df`, then,
    with `num_bootstrap` replicates, in `bootstrap_fits_df`, their summary
    columns (see `tgs_summarize_bootstrap`).
    """
    estimates_df = pd.concat(
        [
            counts_df,
            point_estimate_df[
                [
                    "gain_rate",
                    "selection",
                    "gain_ratio",
                    "selection_lrt_naive",
                    "gain_ratio_lrt_naive",
                ]
            ],
        ],
        axis=1,
    )
    if not num_bootstrap:
        return estimates_df
    return pd.concat(
        [
            estimates_df,
            tgs_summarize_bootstrap(
                point_estimate_df=estimates_df,
                bootstrap_fits_df=bootstrap_fits_df,
            ),
        ],
        axis=1,
    )
