import typing
import warnings

import pandas as pd


def tgs_warn_excluded_replicates(
    is_excluded: pd.Series,
    *,
    trait_names: typing.Sequence[str],
    num_replicates: int,
    reason: str,
) -> None:
    """Implementation detail for `tgs_summarize_bootstrap`.

    Here, tgs stands for trait gain/selection.

    Warns, if any, how many bootstrap replicates were excluded per trait,
    from `is_excluded`, a boolean series indexed by replicate and trait
    index, for `reason`, e.g., "no carriers of the trait".
    """
    num_excluded = is_excluded.groupby(level="trait_index").sum()
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
