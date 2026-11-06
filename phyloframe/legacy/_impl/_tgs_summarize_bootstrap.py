import numpy as np
import polars as pl

from ..._auxlib._calc_two_sided_normal_p import calc_two_sided_normal_p
from ._tgs_warn_excluded_replicates import tgs_warn_excluded_replicates


def tgs_summarize_bootstrap(
    *,
    point_estimate_df: pl.DataFrame,
    bootstrap_fits_df: pl.DataFrame,
) -> pl.DataFrame:
    """Implementation detail for `alifestd_tgs_pendantmdc_estimate_polars`.
    Acronym tgs stands for trait gain/selection.

    Returns bootstrap standard errors, z-scores, and p-values, given point
    estimates and replicate fits with columns `replicate` and `trait_index`.
    """
    # replicates resampling no carriers of a trait leave its selection
    # coefficient undefined, and those resampling no carriers of its neutral
    # reference leave its gain ratio undefined, so exclude them; gain ratio is
    # also undefined, by design, without neutral traits, which does not make a
    # replicate degenerate
    trait_index = bootstrap_fits_df["trait_index"].to_numpy()
    selection_coef = bootstrap_fits_df["selection_coef"].to_numpy()
    gain_ratio = bootstrap_fits_df["gain_ratio"].to_numpy()
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
        tgs_warn_excluded_replicates(
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
