import typing

import numpy as np
import polars as pl
from scipy import sparse

from ._tgs_calc_trait_gain_ratio_lrts import tgs_calc_trait_gain_ratio_lrts
from ._tgs_calc_trait_selection_lrts import tgs_calc_trait_selection_lrts
from ._tgs_fit import tgs_fit_trait
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
    `tgs_merge_episodes`), and, if given, their `survival_shares` over
    `distinct_survival_durations` (see `tgs_calc_nll`). Returns one row per
    trait, in trait index order; see Output Schema.

    Output Schema
    -------------
    'gain_rate' : float
        Fitted gain rate, per unit branch length; 0 without gains.
    'selection_coef' : float
        Fitted selection coefficient; nan without gains.
    'selection_coef_lrt_naive' : float
        Likelihood-ratio statistic for selection coefficient = 0; see
        `tgs_calc_trait_selection_lrts`.
    'gain_ratio' : float
        Gain rate relative to the mean gain rate of the trait's neutral
        reference; see `tgs_normalize_gain_rates_to_neutral`.
    'gain_ratio_lrt_naive' : float
        Likelihood-ratio statistic for gain ratio = 1; see
        `tgs_calc_trait_gain_ratio_lrts`.
    """
    traits = partition_df["trait"].to_numpy()
    trait_rows = [
        np.flatnonzero(traits == trait) for trait in range(num_traits)
    ]
    # each trait's `tgs_calc_nll` arguments, but the rate parameters
    trait_nll_kwargs = [
        dict(
            partitions=partition_df[rows],
            growth_exponent_bound=growth_exponent_bound,
            survival_shares=(
                None if survival_shares is None else survival_shares[rows]
            ),
            distinct_survival_durations=distinct_survival_durations,
        )
        for rows in trait_rows
    ]
    fits_df = pl.DataFrame(
        [
            tgs_fit_trait(
                nll_kwargs=nll_kwargs,
                loss_rate=loss_rate,
                selection_coef_starts=selection_coef_starts,
            )
            for nll_kwargs in trait_nll_kwargs
        ],
        schema={"gain_rate": pl.Float64, "selection_coef": pl.Float64},
    )
    # normalize each gain rate by the mean gain rate of its neutral reference;
    # undefined without neutral reference traits, or if none gained the trait
    fits_df = fits_df.with_columns(
        trait=pl.int_range(pl.len(), dtype=pl.Int64),
        gain_ratio=tgs_normalize_gain_rates_to_neutral(
            gain_rates=fits_df["gain_rate"].to_numpy(),
            neutral_traits=neutral_traits,
        ),
    )
    lrt_kwargs = dict(
        fits_df=fits_df, trait_nll_kwargs=trait_nll_kwargs, loss_rate=loss_rate
    )
    return (
        fits_df.join(
            tgs_calc_trait_selection_lrts(**lrt_kwargs),
            on="trait",
            maintain_order="left",
        )
        .join(
            tgs_calc_trait_gain_ratio_lrts(
                **lrt_kwargs, selection_coef_starts=selection_coef_starts
            ),
            on="trait",
            maintain_order="left",
        )
        .select(
            "gain_rate",
            "selection_coef",
            "selection_coef_lrt_naive",
            "gain_ratio",
            "gain_ratio_lrt_naive",
        )
    )
