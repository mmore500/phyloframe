import numpy as np
import pandas as pd

from ._tgs_extract_trait_masks import tgs_extract_trait_masks


def tgs_count_traits(
    *,
    exposure_df: pd.DataFrame,
    partitions: np.ndarray,
    num_partitions: int,
    weights: np.ndarray,
) -> pd.DataFrame:
    """Implementation detail for `alifestd_tgs_pendantmdc_estimate_polars` and
    `alifestd_tgs_allmdc_estimate_polars`. Acronym tgs stands for trait
    gain/selection.

    Returns, indexed by (`trait`, `partition`), each trait's (weighted)
    count of exposure episodes in each partition, `num_trials`, of their
    gains, `num_successes`, and their summed branch length, `length_sum`,
    given each row of `exposure_df` (see `tgs_calc_exposure_df`), its
    `partitions` index, and its `weights` (e.g., bootstrap draw counts).
    """
    starts_absent, gains = tgs_extract_trait_masks(exposure_df)
    weights = np.expand_dims(np.asarray(weights, dtype=np.float64), 1)
    lengths = np.expand_dims(exposure_df["branch_length"].to_numpy(), 1)

    def count(values: np.ndarray) -> np.ndarray:
        # trait-major, to match the index
        return np.ravel(
            [
                np.bincount(partitions, column, minlength=num_partitions)
                for column in values.T
            ],
        )

    return pd.DataFrame(
        {
            "num_trials": count(starts_absent * weights),
            "num_successes": count(gains * weights),
            "length_sum": count(starts_absent * weights * lengths),
        },
        index=pd.MultiIndex.from_product(
            [range(starts_absent.shape[1]), range(num_partitions)],
            names=["trait", "partition"],
        ),
    )
