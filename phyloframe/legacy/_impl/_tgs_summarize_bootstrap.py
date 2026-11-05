import numpy as np
import pandas as pd

from ..._auxlib._calc_two_sided_normal_p import calc_two_sided_normal_p
from ._tgs_warn_excluded_replicates import tgs_warn_excluded_replicates


def tgs_summarize_bootstrap(
    *,
    point_df: pd.DataFrame,
    replicate_fits_df: pd.DataFrame,
) -> pd.DataFrame:
    """Implementation detail for
    `alifestd_tgs_pendantmdc_estimate_asexual`.

    Here, tgs stands for trait gain/selection.

    Returns bootstrap standard errors, z-scores, and p-values, given point
    estimates and replicate fits indexed by replicate and trait index.
    """
    # replicates resampling no carriers of a trait leave its selection
    # undefined, and those resampling no carriers of its neutral reference
    # leave its gain ratio undefined, so exclude them; gain ratio is also
    # undefined, by design, without neutral traits, which does not make a
    # replicate degenerate
    trait_index = replicate_fits_df.index.get_level_values("trait_index")
    selection = replicate_fits_df["selection"]
    gain_ratio = replicate_fits_df["gain_ratio"]
    is_selection_degenerate = ~np.isfinite(selection)
    is_gain_ratio_degenerate = is_selection_degenerate | (
        np.isfinite(point_df["gain_ratio"].to_numpy())[trait_index]
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
            trait_names=point_df["trait"],
            num_replicates=replicate_fits_df.index.get_level_values(
                0
            ).nunique(),
            reason=reason,
        )

    # bootstrap standard errors (se), i.e., standard deviations across
    # replicates, per trait; nan with fewer than two non-excluded replicates
    selection_se = (
        selection.mask(is_selection_degenerate)
        .groupby(trait_index)
        .std()
        .to_numpy()
    )
    log_gain_ratio_se = (
        np.log(gain_ratio.mask(is_gain_ratio_degenerate))
        .groupby(trait_index)
        .std()
        .to_numpy()
    )
    selection_z = point_df["selection"].to_numpy() / selection_se
    gain_ratio_z = (
        np.log(point_df["gain_ratio"].to_numpy()) / log_gain_ratio_se
    )
    return pd.DataFrame(
        {
            "selection_se": selection_se,
            "selection_z": selection_z,
            "selection_p": calc_two_sided_normal_p(selection_z),
            "log_gain_ratio_se": log_gain_ratio_se,
            "gain_ratio_z": gain_ratio_z,
            "gain_ratio_p": calc_two_sided_normal_p(gain_ratio_z),
            "num_bootstrap_degenerate": is_gain_ratio_degenerate.groupby(
                trait_index
            )
            .sum()
            .to_numpy(),
        },
        index=point_df.index,
    )
