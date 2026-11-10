import numpy as np
import polars as pl


def simulate_exposure_df(
    num_branches: int = 40_000, seed: int = 1, num_neutral: int = 2
) -> pl.DataFrame:
    """Branches, as from `tgs_build_exposure_df_polars`, in one calibration
    bin: half leaves (survival duration
    0), half inner branches of clades of 2 to 9 leaves, every branch at
    risk of each trait and gaining it under the full model.

    Trait 0 has twice the neutral gain rate and selection -0.05; the
    `num_neutral` traits after it are neutral.
    """
    rng = np.random.default_rng(seed)
    is_leaf = np.arange(num_branches) % 2 == 0
    lengths = rng.uniform(0.5, 60, num_branches)
    survival = np.where(is_leaf, 0.0, rng.uniform(0.5, 60, num_branches))
    columns = {
        "origin_time": lengths + survival + rng.uniform(0, 10, num_branches),
        "episode_length": lengths,
        "survival_duration": survival,
        "clade_size": np.where(is_leaf, 1, rng.integers(2, 10, num_branches)),
        "partition_id": np.zeros(num_branches, dtype=np.int64),
        "bin_id": np.zeros(num_branches, dtype=np.int64),
    }
    for i, (gain, selection) in enumerate(
        [(2e-3, -0.05)] + [(1e-3, 0.0)] * num_neutral
    ):
        if selection:
            exposure = np.expm1(selection * lengths) / selection
        else:
            exposure = lengths
        p = gain * exposure * np.exp(selection * survival)
        columns[f"trait{i}_starts_absent"] = np.ones(num_branches, dtype=bool)
        columns[f"trait{i}_gain"] = rng.random(num_branches) < p
    return pl.DataFrame(columns)
