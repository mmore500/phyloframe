import typing

import pandas as pd

from ._tgs_summarize_bootstrap import tgs_summarize_bootstrap


def tgs_append_bootstrap_summary(
    res: pd.DataFrame,
    *,
    replicate_fits: typing.Optional[pd.DataFrame],
    num_bootstrap: int,
) -> pd.DataFrame:
    """Implementation detail for
    `alifestd_tgs_pendantmdc_estimate_asexual` and
    `alifestd_tgs_allmdc_estimate_asexual`.

    Here, tgs stands for trait gain/selection.

    Returns point estimates `res` with bootstrap summary columns (see
    `tgs_summarize_bootstrap`) appended, or `res` unchanged without
    bootstrap replicates. Without `replicate_fits`, i.e., with nothing to
    fit, every replicate matches the point fit.
    """
    if not num_bootstrap:
        return res
    if replicate_fits is None:
        replicate_fits = pd.concat(
            [res] * num_bootstrap,
            keys=range(num_bootstrap),
            names=["replicate", "trait_index"],
        )
    return pd.concat(
        [
            res,
            tgs_summarize_bootstrap(point=res, replicate_fits=replicate_fits),
        ],
        axis=1,
    )
