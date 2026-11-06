import typing

import polars as pl

from ..._auxlib._identity import identity
from ._tgs_calc_trait_gain_ratio_lrts import tgs_calc_trait_gain_ratio_lrts
from ._tgs_calc_trait_selection_lrts import tgs_calc_trait_selection_lrts
from ._tgs_fit import tgs_fit_trait
from ._tgs_normalize_gain_rates_to_neutral import (
    tgs_normalize_gain_rates_to_neutral,
)


def tgs_fit_trait_partitions(
    *,
    trait_nll_kwargs: typing.Sequence[typing.Dict[str, typing.Any]],
    neutral_traits: typing.Sequence[int],
    loss_rate: float,
    selection_coef_starts: typing.Sequence[float],
    progress_wrap: typing.Callable = identity,
) -> pl.DataFrame:
    """Implementation detail for `alifestd_tgs_pendantmdc_estimate_polars` and
    `alifestd_tgs_allmdc_estimate_polars`. Acronym tgs stands for trait
    gain/selection.

    Fits each trait's gain rate and selection coefficient, given each
    trait's `tgs_calc_nll` arguments but the rate parameters,
    `trait_nll_kwargs`, e.g., its partitions from `tgs_merge_episodes` (see
    Input Schema of `tgs_calc_nll`).

    Fits one trait at a time, wrapped in `progress_wrap` (e.g., tqdm).
    Returns one row per trait, in trait index order; see Output Schema.

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
    fits_df = pl.DataFrame(
        [
            tgs_fit_trait(
                nll_kwargs=nll_kwargs,
                loss_rate=loss_rate,
                selection_coef_starts=selection_coef_starts,
            )
            for nll_kwargs in progress_wrap(trait_nll_kwargs)
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
