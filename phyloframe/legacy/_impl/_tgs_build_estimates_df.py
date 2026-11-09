import warnings

import polars as pl


def _tgs_warn_undefined_estimates(estimates_df: pl.DataFrame) -> None:
    """Implementation detail for `tgs_build_estimates_df`.

    Warns of traits whose estimates are undefined (nan): without gains,
    the selection coefficient and likelihood-ratio statistics; without
    gains of the neutral reference, the gain ratio and its statistic.
    """
    no_gains = pl.col("selection_coef").is_nan()
    traits_without_gains = estimates_df.filter(no_gains)["trait"].to_list()
    if traits_without_gains:
        warnings.warn(
            f"traits {traits_without_gains} have no gains, so their "
            "selection coefficients and likelihood-ratio statistics are "
            "undefined (nan)",
        )
    traits_without_reference = estimates_df.filter(
        ~no_gains & pl.col("gain_ratio").is_nan()
    )["trait"].to_list()
    if traits_without_reference:
        warnings.warn(
            f"traits {traits_without_reference} have neutral references "
            "without gains, so their gain ratios and gain ratio "
            "likelihood-ratio statistics are undefined (nan)",
        )


def tgs_build_estimates_df(
    episode_tallies_df: pl.DataFrame,
    *,
    fits_df: pl.DataFrame,
) -> pl.DataFrame:
    """Implementation detail for `alifestd_tgs_estimate_polars`. Acronym tgs
    stands for trait gain/selection.

    Returns one row per trait, concatenating `episode_tallies_df`'s per-trait
    exposure episode and gain tallies (see `tgs_tally_exposure_episodes`)
    and `fits_df`'s estimates (see `tgs_fit_traits`); see Output Schema.
    Warns of traits whose estimates are undefined (nan), e.g., traits
    without gains.

    Output Schema
    -------------
    'trait' : string
        Trait name.
    'num_leaf_episodes' : integer
        Leaf exposure episodes, i.e., leaf branches that started without
        the trait.
    'num_leaf_gains' : integer
        Of those, the number that gained the trait.
    'num_inner_episodes' : integer
        Inner exposure episodes, i.e., inner branches that started without
        the trait.
    'num_inner_gains' : integer
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

    """
    estimates_df = pl.concat(
        [
            episode_tallies_df,
            fits_df.select(
                "gain_rate",
                "selection_coef",
                "gain_ratio",
                "selection_coef_lrt_naive",
                "gain_ratio_lrt_naive",
            ),
        ],
        how="horizontal",
    )
    _tgs_warn_undefined_estimates(estimates_df)
    return estimates_df
