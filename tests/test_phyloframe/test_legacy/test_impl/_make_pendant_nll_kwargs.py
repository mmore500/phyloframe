import typing

import numpy as np
import polars as pl
from scipy import sparse


def make_pendant_nll_kwargs(
    coarsened_df: pl.DataFrame,
) -> typing.Dict[str, typing.Any]:
    """`tgs_calc_nll` arguments but the rate parameters for `coarsened_df`
    of leaf pendant branches, whose survival duration is 0."""
    return dict(
        coarsened_df=coarsened_df,
        survival_pool_mass=sparse.csr_matrix(np.ones((len(coarsened_df), 1))),
        distinct_survival_durations=np.zeros(1),
    )
