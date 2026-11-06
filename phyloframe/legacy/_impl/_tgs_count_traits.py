import typing

import numpy as np
import pandas as pd


def tgs_count_traits(
    *,
    partitions: np.ndarray,
    num_partitions: int,
    lengths: np.ndarray,
    weights: np.ndarray,
    trait_episodes: typing.Sequence[np.ndarray],
    trait_gains: typing.Sequence[np.ndarray],
) -> typing.List[pd.DataFrame]:
    """Implementation detail for `alifestd_tgs_pendantmdc_estimate_polars` and
    `alifestd_tgs_allmdc_estimate_polars`. Acronym tgs stands for trait
    gain/selection.

    Returns, per trait, a DataFrame with a row per partition and columns
    `num_trials`, the (weighted) count of the trait's exposure episodes
    there (positions `trait_episodes`, into `partitions`), `num_successes`,
    the count of its gains (positions `trait_gains`), and `length_sum`, the
    episodes' (weighted) summed length.
    """

    def count(positions: np.ndarray, values: np.ndarray) -> np.ndarray:
        return np.bincount(
            partitions[positions], values[positions], minlength=num_partitions
        )

    return [
        pd.DataFrame(
            {
                "num_trials": count(episodes, weights),
                "num_successes": count(gains, weights),
                "length_sum": count(episodes, weights * lengths),
            },
        )
        for episodes, gains in zip(trait_episodes, trait_gains)
    ]
