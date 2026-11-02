import typing

import pandas as pd

from ._tgs_summarize_bootstrap import tgs_summarize_bootstrap


def tgs_build_estimates_df(
    estimates_df: pd.DataFrame,
    *,
    replicate_fits_df: typing.Optional[pd.DataFrame],
    num_bootstrap: int,
) -> pd.DataFrame:
    """Implementation detail for
    `alifestd_tgs_pendantmdc_estimate_asexual` and
    `alifestd_tgs_allmdc_estimate_asexual`.

    Here, tgs stands for trait gain/selection.

    Returns point estimates `estimates_df` with bootstrap summary columns (see
    `tgs_summarize_bootstrap`) appended, or `estimates_df` unchanged without
    bootstrap replicates. Without `replicate_fits_df`, i.e., with nothing to
    fit, every replicate matches the point fit.
    """
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
