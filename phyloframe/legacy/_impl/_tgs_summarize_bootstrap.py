import typing
import warnings

import numpy as np
import polars as pl

from ..._auxlib._calc_two_sided_normal_p import calc_two_sided_normal_p


def _warn_excluded_replicates(
    is_excluded: np.ndarray,
    *,
    trait_index: np.ndarray,
    trait_names: typing.Sequence[str],
    num_replicates: int,
    reason: str,
) -> None:
    """Implementation detail for `tgs_summarize_bootstrap`.

    Warns, if any, how many bootstrap replicates were excluded per trait,
    from `is_excluded`, a boolean array over replicate fits with trait
    indices `trait_index`, for `reason`, e.g., "no carriers of the trait".
    """
    num_excluded = np.bincount(
        trait_index, weights=is_excluded, minlength=len(trait_names)
    ).astype(np.int64)
    if num_excluded.any():
        counts = ", ".join(
            f"{trait} ({count} of {num_replicates})"
            for trait, count in zip(trait_names, num_excluded)
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
    """Implementation detail for `alifestd_tgs_pendantmdc_bootstrap_polars`
    and `alifestd_tgs_allmdc_bootstrap_polars`. Acronym tgs stands for trait
    gain/selection.

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
    trait_index = bootstrap_fits_df["trait_index"].to_numpy()
    selection_coef = bootstrap_fits_df["selection_coef"].to_numpy()
    gain_ratio = bootstrap_fits_df["gain_ratio"].to_numpy()
    # exclude undefined estimates, e.g., the selection coefficient of a
    # replicate resampling no carriers of the trait, or the gain ratio of one
    # resampling no carriers of its neutral reference; a gain ratio undefined
    # in the point estimate too does not make a replicate degenerate
    is_selection_degenerate = ~np.isfinite(selection_coef)
    is_gain_ratio_degenerate = is_selection_degenerate | (
        np.isfinite(point_estimate_df["gain_ratio"].to_numpy())[trait_index]
        & ~np.isfinite(gain_ratio)
    )
    for is_excluded, reason in (
        (is_selection_degenerate, "no carriers of the trait"),
        (
            is_gain_ratio_degenerate & ~is_selection_degenerate,
            "no carriers of the trait's neutral reference",
        ),
    ):
        _warn_excluded_replicates(
            is_excluded,
            trait_index=trait_index,
            trait_names=point_estimate_df["trait"],
            num_replicates=bootstrap_fits_df["replicate"].n_unique(),
            reason=reason,
        )

    # per trait, bootstrap standard errors (se), i.e., standard deviations
    # across non-excluded replicates (nan with fewer than two), and the
    # number of degenerate replicates
    trait_bootstrap_stats_df = (
        pl.DataFrame(
            {
                "trait_index": trait_index,
                "selection_coef": np.where(
                    is_selection_degenerate, np.nan, selection_coef
                ),
                "log_gain_ratio": np.log(
                    np.where(is_gain_ratio_degenerate, np.nan, gain_ratio)
                ),
                "is_gain_ratio_degenerate": is_gain_ratio_degenerate,
            },
        )
        .group_by("trait_index")
        .agg(
            selection_coef_se=pl.col("selection_coef").drop_nans().std(),
            log_gain_ratio_se=pl.col("log_gain_ratio").drop_nans().std(),
            num_bootstrap_degenerate=pl.col("is_gain_ratio_degenerate")
            .cast(pl.Int64)
            .sum(),
        )
        .sort("trait_index")
    )
    selection_coef_se = trait_bootstrap_stats_df[
        "selection_coef_se"
    ].to_numpy()
    log_gain_ratio_se = trait_bootstrap_stats_df[
        "log_gain_ratio_se"
    ].to_numpy()
    selection_coef_z = (
        point_estimate_df["selection_coef"].to_numpy() / selection_coef_se
    )
    gain_ratio_z = (
        np.log(point_estimate_df["gain_ratio"].to_numpy()) / log_gain_ratio_se
    )
    return pl.DataFrame(
        {
            "selection_coef_se": selection_coef_se,
            "selection_coef_z": selection_coef_z,
            "selection_coef_p": calc_two_sided_normal_p(selection_coef_z),
            "log_gain_ratio_se": log_gain_ratio_se,
            "gain_ratio_z": gain_ratio_z,
            "gain_ratio_p": calc_two_sided_normal_p(gain_ratio_z),
            "num_bootstrap_degenerate": trait_bootstrap_stats_df[
                "num_bootstrap_degenerate"
            ],
        },
    )
