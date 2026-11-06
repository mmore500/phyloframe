import functools
import typing

import numpy as np
import polars as pl
from scipy import sparse

from ._tgs_calc_gain_ratio_lrts import tgs_calc_gain_ratio_lrts
from ._tgs_calc_nll import tgs_calc_nll
from ._tgs_fit import TgsFitResult, tgs_fit_partitions
from ._tgs_normalize_gain_rates_to_neutral import (
    tgs_normalize_gain_rates_to_neutral,
)


def tgs_fit_trait_partitions(
    *,
    partition_df: pl.DataFrame,
    num_traits: int,
    neutral_traits: typing.Sequence[int],
    loss_rate: float,
    selection_coef_starts: typing.Sequence[float],
    growth_exponent_bound: float,
    survival_shares: typing.Optional[sparse.csr_matrix] = None,
    distinct_survival_durations: typing.Optional[np.ndarray] = None,
) -> pl.DataFrame:
    """Implementation detail for `alifestd_tgs_pendantmdc_estimate_polars` and
    `alifestd_tgs_allmdc_estimate_polars`. Acronym tgs stands for trait
    gain/selection.

    Fits each of `num_traits` traits' gain rate and selection coefficient to
    its partitions, the rows of `partition_df` with its `trait` (see
    `tgs_merge_episodes`),
    and, if given, their `survival_shares` over `distinct_survival_durations`
    (see `tgs_calc_nll`). Returns one row per trait, with columns of
    `TgsFitResult` (see `tgs_fit_partitions`) plus `gain_ratio`, the gain
    rate relative to neutral traits, and `gain_ratio_lrt_naive` (see `tgs_calc_gain_ratio_lrts`).
    """
    traits = partition_df["trait"].to_numpy()
    trait_rows = [
        np.flatnonzero(traits == trait) for trait in range(num_traits)
    ]
    trait_partitions = [partition_df[rows] for rows in trait_rows]
    calc_nlls = [
        functools.partial(
            tgs_calc_nll,
            partitions=partitions,
            growth_exponent_bound=growth_exponent_bound,
            survival_shares=(
                None if survival_shares is None else survival_shares[rows]
            ),
            distinct_survival_durations=distinct_survival_durations,
        )
        for partitions, rows in zip(trait_partitions, trait_rows)
    ]
    fits = [
        tgs_fit_partitions(
            calc_nll=calc_nll,
            partitions=partitions,
            loss_rate=loss_rate,
            selection_coef_starts=selection_coef_starts,
        )
        for partitions, calc_nll in zip(trait_partitions, calc_nlls)
    ]
    res = pl.DataFrame(
        {
            field: [getattr(fit, field) for fit in fits]
            for field in TgsFitResult._fields
        },
        schema={field: pl.Float64 for field in TgsFitResult._fields},
    )
    # normalize each gain rate by the mean gain rate of its neutral reference;
    # undefined without neutral reference traits, or if none gained the trait
    res = res.with_columns(
        gain_ratio=tgs_normalize_gain_rates_to_neutral(
            gain_rates=res["gain_rate"].to_numpy(),
            neutral_traits=neutral_traits,
        ),
    )
    return res.with_columns(
        gain_ratio_lrt_naive=tgs_calc_gain_ratio_lrts(
            fits_df=res,
            trait_partitions=trait_partitions,
            calc_nlls=calc_nlls,
            loss_rate=loss_rate,
            selection_coef_starts=selection_coef_starts,
        ),
    )
