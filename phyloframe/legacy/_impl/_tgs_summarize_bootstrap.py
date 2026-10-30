import warnings

import numpy as np
import pandas as pd

from ..._auxlib._calc_two_sided_normal_p import calc_two_sided_normal_p


def tgs_summarize_bootstrap(
    *,
    point: pd.DataFrame,
    replicate_fits: pd.DataFrame,
) -> pd.DataFrame:
    """Implementation detail for
    `alifestd_estimate_trait_gain_selection_pendant_asexual`.

    Here, tgs stands for trait gain/selection.

    Returns bootstrap standard errors, z-scores, and p-values, given point
    estimates and replicate fits indexed by replicate and trait index.
    """
    # replicates resampling no carriers of a trait leave its selection
    # undefined, and those resampling no carriers of its neutral reference
    # leave its gain ratio undefined, so exclude them; gain ratio is also
    # undefined, by design, without neutral traits, which does not make a
    # replicate degenerate
    trait_index = replicate_fits.index.get_level_values("trait_index")
    selection = replicate_fits["selection"]
    gain_ratio = replicate_fits["gain_ratio"]
    is_selection_degenerate = ~np.isfinite(selection)
    is_gain_ratio_degenerate = is_selection_degenerate | (
        np.isfinite(point["gain_ratio"].to_numpy())[trait_index]
        & ~np.isfinite(gain_ratio)
    )
    num_replicates = replicate_fits.index.get_level_values(0).nunique()
    for is_excluded, reason in (
        (is_selection_degenerate, "no carriers of the trait"),
        (
            is_gain_ratio_degenerate & ~is_selection_degenerate,
            "no carriers of the trait's neutral reference",
        ),
    ):
        num_excluded = is_excluded.groupby(trait_index).sum()
        if num_excluded.any():
            counts = ", ".join(
                f"{trait} ({count} of {num_replicates})"
                for trait, count in zip(point["trait"], num_excluded)
                if count
            )
            warnings.warn(
                f"bootstrap replicates that resampled {reason} were excluded "
                f"for traits {counts}, so their standard errors may be "
                "underestimated.",
            )

    # standard deviation across replicates, per trait; nan with fewer than
    # two non-excluded replicates
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
    res = pd.DataFrame(
        {
            "selection_se": selection_se,
            "selection_z": point["selection"] / selection_se,
            "log_gain_ratio_se": log_gain_ratio_se,
            "gain_ratio_z": np.log(point["gain_ratio"]) / log_gain_ratio_se,
            "num_bootstrap_degenerate": is_gain_ratio_degenerate.groupby(
                trait_index
            )
            .sum()
            .to_numpy(),
        },
    )
    res["selection_p"] = calc_two_sided_normal_p(res["selection_z"])
    res["gain_ratio_p"] = calc_two_sided_normal_p(res["gain_ratio_z"])
    return res[
        [
            "selection_se",
            "selection_z",
            "selection_p",
            "log_gain_ratio_se",
            "gain_ratio_z",
            "gain_ratio_p",
            "num_bootstrap_degenerate",
        ]
    ]
