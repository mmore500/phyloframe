import typing

import opytional as opyt
import polars as pl

from ._tgs_summarize_bootstrap import tgs_summarize_bootstrap


def tgs_build_estimates_df(
    counts_df: pl.DataFrame,
    *,
    point_estimate_df: pl.DataFrame,
    bootstrap_fits_df: typing.Optional[pl.DataFrame],
) -> pl.DataFrame:
    """Implementation detail for `alifestd_tgs_pendantmdc_estimate_polars` and
    `alifestd_tgs_allmdc_estimate_polars`. Acronym tgs stands for trait
    gain/selection.

    Returns one row per trait, joining `counts_df`'s per-trait counts,
    `point_estimate_df`'s estimates, and bootstrap summaries of
    `bootstrap_fits_df`; see Output Schema.

    Output Schema
    -------------
    'trait' : string
        Trait name.
    'num_leaves' : integer
        Leaf exposure episodes, i.e., leaf branches that started without
        the trait.
    'num_trait_present' : integer
        Of those, the number that gained the trait.
    'num_inner_branches' : integer, allmdc only
        Inner exposure episodes, i.e., inner branches that started without
        the trait.
    'num_inner_gains' : integer, allmdc only
        Of those, the number that gained the trait.
    'gain_rate' : float
        Estimated rate of trait gain, per unit branch length.
    'selection_coef' : float
        Estimated selection coefficient.
    'gain_ratio' : float
        Gain rate relative to the mean gain rate of the other neutral
        traits.
    'selection_coef_lrt_naive' : float
        Likelihood-ratio statistic for selection coefficient = 0.
    'gain_ratio_lrt_naive' : float
        Likelihood-ratio statistic for gain ratio = 1.

    With `num_bootstrap` replicates, also:

    'selection_coef_se' : float
        Bootstrap standard error of the selection coefficient.
    'selection_coef_z' : float
        Selection coefficient over its standard error.
    'selection_coef_p' : float
        Two-sided normal p-value of `selection_coef_z`.
    'log_gain_ratio_se' : float
        Bootstrap standard error of the log gain ratio.
    'gain_ratio_z' : float
        Log gain ratio over its standard error.
    'gain_ratio_p' : float
        Two-sided normal p-value of `gain_ratio_z`.
    'num_bootstrap_degenerate' : integer
        Replicates excluded for an undefined selection coefficient or gain
        ratio.
    """
    return pl.concat(
        [
            counts_df,
            point_estimate_df.select(
                "gain_rate",
                "selection_coef",
                "gain_ratio",
                "selection_coef_lrt_naive",
                "gain_ratio_lrt_naive",
            ),
            opyt.apply_if_or_else(
                bootstrap_fits_df,
                lambda df: tgs_summarize_bootstrap(
                    point_estimate_df=point_estimate_df,
                    bootstrap_fits_df=df,
                ),
                pl.DataFrame,
            ),
        ],
        how="horizontal",
    )
