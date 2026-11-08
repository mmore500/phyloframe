import polars as pl


def tgs_build_estimates_df(
    counts_df: pl.DataFrame,
    *,
    point_estimate_df: pl.DataFrame,
) -> pl.DataFrame:
    """Implementation detail for `alifestd_tgs_pendantmdc_estimate_polars` and
    `alifestd_tgs_allmdc_estimate_polars`. Acronym tgs stands for trait
    gain/selection.

    Returns one row per trait, joining `counts_df`'s per-trait counts and
    `point_estimate_df`'s estimates; see Output Schema.

    Output Schema
    -------------
    'trait' : string
        Trait name.
    'num_leaf_episodes' : integer
        Leaf exposure episodes, i.e., leaf branches that started without
        the trait.
    'num_leaf_gains' : integer
        Of those, the number that gained the trait.
    'num_inner_episodes' : integer, allmdc only
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
        ],
        how="horizontal",
    )
