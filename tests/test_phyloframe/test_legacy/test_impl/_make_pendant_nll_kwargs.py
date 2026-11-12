import typing

import numpy as np
import polars as pl
from scipy import sparse


def make_pendant_nll_kwargs(
    coarsegrain_df: pl.DataFrame,
) -> typing.Dict[str, typing.Any]:
    """`tgs_calc_nll` arguments but the rate parameters for `coarsegrain_df`
    of leaf pendant branches, whose survival duration is 0."""
    return dict(
        coarsegrain_df=coarsegrain_df,
        survival_duration_granule_mass=sparse.csr_matrix(
            np.ones((len(coarsegrain_df), 1))
        ),
        distinct_survival_durations=np.zeros(1),
    )
