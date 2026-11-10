import functools
import typing

import numpy as np
import polars as pl


def tgs_count_trait_bin_exposures(
    exposure_df: pl.DataFrame,
    *,
    trait_ids: typing.Sequence[int],
    weights: np.ndarray,
) -> pl.DataFrame:
    """Implementation detail for `alifestd_tgs_estimate_polars`.

    Returns exposure counts of each of `trait_ids` in each calibration bin
    of `exposure_df` (see `tgs_build_exposure_df_polars`), with episodes
    weighted by `weights` (e.g., bootstrap draw counts).

    Output Schema
    -------------
    'trait_id' : integer
        Trait index.
    'bin_id' : integer
        Calibration bin index; every bin, for each trait.
    'num_trials' : float
        Weighted count of the trait's exposure episodes in the bin.
    'num_successes' : float
        Of those, the weighted count that gained the trait.
    'length_sum' : float
        Weighted sum of those episodes' lengths.
    """
    episode_lengths = exposure_df["episode_length"].to_numpy()
    assert np.isfinite(episode_lengths).all()
    bin_ids = exposure_df["bin_id"].to_numpy()
    num_bins = bin_ids.max() + 1  # bins are numbered densely
    sum_by_bin = functools.partial(np.bincount, bin_ids, minlength=num_bins)

    def count_trait_bin_exposures(trait_id: int) -> pl.DataFrame:
        trait_weights = (
            weights * exposure_df[f"trait{trait_id}_starts_absent"].to_numpy()
        )
        return pl.DataFrame(
            {
                "trait_id": np.full(num_bins, trait_id, dtype=np.int64),
                "bin_id": np.arange(num_bins, dtype=np.int64),
                "num_trials": sum_by_bin(trait_weights),
                "num_successes": sum_by_bin(
                    trait_weights
                    * exposure_df[f"trait{trait_id}_gain"].to_numpy()
                ),
                "length_sum": sum_by_bin(trait_weights * episode_lengths),
            },
        )

    return pl.concat(map(count_trait_bin_exposures, trait_ids))
