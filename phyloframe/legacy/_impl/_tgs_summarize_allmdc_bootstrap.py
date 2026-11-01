import typing

import numpy as np
import pandas as pd

from ._tgs_summarize_bootstrap import tgs_summarize_bootstrap


def tgs_summarize_allmdc_bootstrap(
    *,
    res: pd.DataFrame,
    replicate_fits: typing.Sequence[typing.Tuple[pd.DataFrame, np.ndarray]],
    shift_columns: typing.Sequence[typing.Optional[str]],
    num_bootstrap: int,
) -> pd.DataFrame:
    """Implementation detail for `alifestd_tgs_allmdc_estimate_asexual`.

    Here, tgs stands for trait gain/selection.

    Appends bootstrap standard errors, z-scores, and p-values (see
    `tgs_summarize_bootstrap`) to point estimates `res`, as well as a
    standard error column `{shift}_se` per clade-size shift. Replicates are
    (fits, shifts), with shifts aligned to `shift_columns` (None for the
    first, single-leaf group). Without replicates (nothing to fit), every
    replicate is taken to match `res`.
    """
    if replicate_fits:
        fits = pd.concat(
            [fit for fit, __ in replicate_fits],
            keys=range(num_bootstrap),
            names=["replicate", "trait_index"],
        )
    else:
        fits = pd.concat(
            [res] * num_bootstrap,
            keys=range(num_bootstrap),
            names=["replicate", "trait_index"],
        )
    res = pd.concat(
        [res, tgs_summarize_bootstrap(point=res, replicate_fits=fits)], axis=1
    )
    shift_se = {
        f"{column}_se": np.nan
        for column in res.columns
        if column.startswith("clade_size_shift_")
    }
    for j, column in enumerate(shift_columns):
        if column is None or not replicate_fits:
            continue
        values = np.array([shifts[j] for __, shifts in replicate_fits])
        if np.isfinite(values).sum() >= 2:
            shift_se[f"{column}_se"] = np.nanstd(values, ddof=1)
    return res.assign(**shift_se)
