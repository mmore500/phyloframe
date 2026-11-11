import functools
import typing

import numpy as np
import polars as pl


def tgs_count_trait_exposures_binwise(
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
    'num_episodes' : float
        Weighted count of the trait's exposure episodes in the bin.
    'num_gains' : float
        Of those, the weighted count that gained the trait.
    'total_episode_duration' : float
        Weighted sum of those episodes' durations.
    """
    episode_durations = exposure_df["episode_duration"].to_numpy()
    assert np.isfinite(episode_durations).all()
    bin_ids = exposure_df["bin_id"].to_numpy()
    num_bins = bin_ids.max() + 1  # bins are numbered densely
    sum_by_bin = functools.partial(np.bincount, bin_ids, minlength=num_bins)

    def count_trait_exposures_binwise(trait_id: int) -> pl.DataFrame:
        trait_weights = (
            weights
            * exposure_df[f"trait{trait_id}_episode_starts_absent"].to_numpy()
        )
        return pl.DataFrame(
            {
                "trait_id": np.full(num_bins, trait_id, dtype=np.int64),
                "bin_id": np.arange(num_bins, dtype=np.int64),
                "num_episodes": sum_by_bin(trait_weights),
                "num_gains": sum_by_bin(
                    trait_weights
                    * exposure_df[f"trait{trait_id}_gain"].to_numpy()
                ),
                "total_episode_duration": sum_by_bin(
                    trait_weights * episode_durations
                ),
            },
        )

    return pl.concat(
        map(count_trait_exposures_binwise, trait_ids), how="vertical"
    )
