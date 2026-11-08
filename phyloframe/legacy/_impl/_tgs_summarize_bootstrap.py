import typing
import warnings

import numpy as np
import polars as pl

from ..._auxlib._calc_two_sided_normal_p import calc_two_sided_normal_p


def _warn_excluded_replicates(
    is_excluded: np.ndarray,
    *,
    trait_ids: np.ndarray,
    trait_names: typing.Sequence[str],
    num_replicates: int,
    reason: str,
) -> None:
    """Implementation detail for `tgs_summarize_bootstrap`.

    Warns, if any, how many bootstrap replicates were excluded per trait,
    from `is_excluded`, a boolean array over replicate fits with trait
    indices `trait_ids`, for `reason`, e.g., "no carriers of the trait".
    """
    num_excluded = np.bincount(
        trait_ids, weights=is_excluded, minlength=len(trait_names)
    ).astype(np.int64)
    if num_excluded.any():
        counts = ", ".join(
            f"{trait_name} ({count} of {num_replicates})"
            for trait_name, count in zip(trait_names, num_excluded)
            if count
        )
        warnings.warn(
            f"bootstrap replicates that resampled {reason} were excluded "
            f"for traits {counts}, so their standard errors may be "
            "underestimated.",
        )


def tgs_summarize_bootstrap(
    *,
    point_estimate_df: pl.DataFrame,
    bootstrap_fits_df: pl.DataFrame,
) -> pl.DataFrame:
    """Implementation detail for `alifestd_tgs_pendantmdc_estimate_polars`.
    Acronym tgs stands for trait gain/selection.

    Returns bootstrap standard errors, z-scores, and p-values, one row per
    trait, given point estimates and replicate fits (see
    `tgs_bootstrap_trait_fits`); see Output Schema.

    Output Schema
    -------------
    'selection_coef_se' : float
        Bootstrap standard error of the selection coefficient, over
        replicates with a defined selection coefficient; nan with fewer
        than two.
    'selection_coef_z' : float
        Selection coefficient over its standard error.
    'selection_coef_p' : float
        Two-sided normal p-value of `selection_coef_z`.
    'log_gain_ratio_se' : float
        Bootstrap standard error of the log gain ratio, over replicates
        with a defined gain ratio; nan with fewer than two.
    'gain_ratio_z' : float
        Log gain ratio over its standard error.
    'gain_ratio_p' : float
        Two-sided normal p-value of `gain_ratio_z`.
    'num_bootstrap_degenerate' : integer
        Replicates excluded for an undefined selection coefficient or gain
        ratio.
    """
    trait_ids = bootstrap_fits_df["trait_id"].to_numpy()
    selection_coef = bootstrap_fits_df["selection_coef"].to_numpy()
    gain_ratio = bootstrap_fits_df["gain_ratio"].to_numpy()
    # exclude undefined estimates, e.g., the selection coefficient of a
    # replicate resampling no carriers of the trait, or the gain ratio of one
    # resampling no carriers of its neutral reference; a gain ratio undefined
    # in the point estimate too does not make a replicate degenerate
    is_selection_degenerate = ~np.isfinite(selection_coef)
    is_gain_ratio_degenerate = is_selection_degenerate | (
        np.isfinite(point_estimate_df["gain_ratio"].to_numpy())[trait_ids]
        & ~np.isfinite(gain_ratio)
    )
    warn_kwargs = dict(
        trait_ids=trait_ids,
        trait_names=point_estimate_df["trait"],
        num_replicates=bootstrap_fits_df["replicate"].n_unique(),
    )
    _warn_excluded_replicates(
        is_selection_degenerate,
        reason="no carriers of the trait",
        **warn_kwargs,
    )
    _warn_excluded_replicates(
        is_gain_ratio_degenerate & ~is_selection_degenerate,
        reason="no carriers of the trait's neutral reference",
        **warn_kwargs,
    )

    # per trait, bootstrap standard errors (se), i.e., standard deviations
    # across non-excluded replicates (nan with fewer than two), and the
    # number of degenerate replicates
    trait_bootstrap_stats_df = (
        pl.DataFrame(
            {
                "trait_id": trait_ids,
                "selection_coef": np.where(
                    is_selection_degenerate, np.nan, selection_coef
                ),
                "log_gain_ratio": np.log(
                    np.where(is_gain_ratio_degenerate, np.nan, gain_ratio)
                ),
                "is_gain_ratio_degenerate": is_gain_ratio_degenerate,
            },
        )
        .group_by("trait_id")
        .agg(
            selection_coef_se=pl.col("selection_coef").drop_nans().std(),
            log_gain_ratio_se=pl.col("log_gain_ratio").drop_nans().std(),
            num_bootstrap_degenerate=pl.col("is_gain_ratio_degenerate")
            .cast(pl.Int64)
            .sum(),
        )
        .sort("trait_id")
    )
    return (
        trait_bootstrap_stats_df.with_columns(
            selection_coef_z=pl.lit(point_estimate_df["selection_coef"])
            / pl.col("selection_coef_se"),
            gain_ratio_z=pl.lit(point_estimate_df["gain_ratio"]).log()
            / pl.col("log_gain_ratio_se"),
        )
        .with_columns(
            selection_coef_p=pl.col("selection_coef_z").map_batches(
                calc_two_sided_normal_p, return_dtype=pl.Float64
            ),
            gain_ratio_p=pl.col("gain_ratio_z").map_batches(
                calc_two_sided_normal_p, return_dtype=pl.Float64
            ),
        )
        .select(
            "selection_coef_se",
            "selection_coef_z",
            "selection_coef_p",
            "log_gain_ratio_se",
            "gain_ratio_z",
            "gain_ratio_p",
            "num_bootstrap_degenerate",
        )
    )
