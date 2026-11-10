import typing
import warnings

import polars as pl

from ..._auxlib._calc_two_sided_normal_p import calc_two_sided_normal_p


def tgs_summarize_bootstrap(
    estimates_df: pl.DataFrame,
    *,
    replicate_fits: typing.Sequence[pl.DataFrame],
) -> pl.DataFrame:
    """Implementation detail for `alifestd_tgs_estimate_polars`.

    Returns bootstrap standard errors, z-scores, and p-values, one row per
    trait, given point estimates and each bootstrap replicate's fits (see
    `tgs_fit_traits`). Warns of traits with replicates excluded for an
    undefined estimate.

    Output Schema
    -------------
    'selection_coef_se' : float
        Standard deviation of the selection coefficient across replicates.
    'selection_coef_z' : float
        Selection coefficient over its standard error.
    'selection_coef_p' : float
        Two-sided normal p-value of `selection_coef_z`.
    'log_gain_ratio_se' : float
        Standard deviation of the log gain ratio across replicates.
    'gain_ratio_z' : float
        Log gain ratio over its standard error.
    'gain_ratio_p' : float
        Two-sided normal p-value of `gain_ratio_z`.
    'num_bootstrap_degenerate' : integer
        Replicates excluded for an undefined selection coefficient or log
        gain ratio, e.g., having resampled no gains of the trait.
    """
    is_selection_defined = pl.col("selection_coef").is_finite()
    log_gain_ratio = pl.col("gain_ratio").log()
    # finite only with gains of both the trait and its neutral reference
    is_gain_ratio_defined = log_gain_ratio.is_finite()

    summary_df = (
        pl.concat(
            fits_df.lazy().with_row_index("trait_id")
            for fits_df in replicate_fits
        )
        .group_by("trait_id")
        .agg(
            selection_coef_se=pl.col("selection_coef")
            .filter(is_selection_defined)
            .std(),
            log_gain_ratio_se=log_gain_ratio.filter(
                is_gain_ratio_defined
            ).std(),
            num_bootstrap_degenerate=(~is_gain_ratio_defined)
            .sum()
            .cast(pl.Int64),
        )
        .sort("trait_id")
        .with_columns(
            selection_coef_z=estimates_df["selection_coef"]
            / pl.col("selection_coef_se"),
            gain_ratio_z=estimates_df["gain_ratio"].log()
            / pl.col("log_gain_ratio_se"),
        )
        .select(
            "selection_coef_se",
            "selection_coef_z",
            pl.col("selection_coef_z")
            .map_batches(calc_two_sided_normal_p, return_dtype=pl.Float64)
            .alias("selection_coef_p"),
            "log_gain_ratio_se",
            "gain_ratio_z",
            pl.col("gain_ratio_z")
            .map_batches(calc_two_sided_normal_p, return_dtype=pl.Float64)
            .alias("gain_ratio_p"),
            "num_bootstrap_degenerate",
        )
        .collect()
    )

    degenerate_traits = estimates_df.filter(
        summary_df["num_bootstrap_degenerate"] > 0
    )["trait"].to_list()
    if degenerate_traits:
        warnings.warn(
            "bootstrap replicates with undefined estimates were excluded for "
            f"traits {degenerate_traits}, so their standard errors may be "
            "underestimated; see num_bootstrap_degenerate.",
        )

    return summary_df
