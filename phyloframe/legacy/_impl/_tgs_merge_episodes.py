import typing

import numpy as np
import pandas as pd
import scipy.sparse as scipy_sparse
import sklearn.preprocessing as sklearn_preprocessing

from ._tgs_extract_trait_masks import tgs_extract_trait_masks


def tgs_merge_episodes(
    *,
    exposure_df: pd.DataFrame,
    partitions: np.ndarray,
    calibration_factors: np.ndarray,
    weights: np.ndarray,
    length_rtol: float,
    survival_duration_indices: np.ndarray,
    num_survival_durations: int,
) -> typing.Tuple[pd.DataFrame, scipy_sparse.csr_matrix]:
    """Implementation detail for `alifestd_tgs_pendantmdc_estimate_polars` and
    `alifestd_tgs_allmdc_estimate_polars`. Acronym tgs stands for trait
    gain/selection.

    Merges each trait's exposure episodes (rows of `exposure_df` starting
    without the trait; see `tgs_calc_exposure_df`) whose calibrated branch
    lengths agree within relative tolerance `length_rtol`, i.e., fall in
    the same bin of a log-scale grid with ratio 1 + `length_rtol` (0 for
    no merging). Each episode's length is calibrated by its partition's
    factor, `calibration_factors[trait, partition]`.

    Returns `partition_df`, one row per trait and merged bin with
    episodes, trait-major, with columns `trait` (index, categorical over
    all traits), the (weighted, by `weights`) counts of episodes,
    `num_trials`, and of gains, `num_successes`, and their mean calibrated
    `length`. Also returns `survival_shares`, a sparse matrix with a row
    per `partition_df` row and a column per distinct survival duration
    (episodes' `survival_duration_indices`), giving the share of the row's
    episodes with each.
    """
    starts_absent, gains = tgs_extract_trait_masks(exposure_df)
    num_traits = starts_absent.shape[1]
    episode_traits, episodes = np.nonzero(starts_absent.T)  # trait-major
    lengths = (
        exposure_df["branch_length"].to_numpy()[episodes]
        * calibration_factors[episode_traits, partitions[episodes]]
    )
    if length_rtol > 0:
        # log-scale grid; zero lengths (from zero calibration factors) get
        # a bin of their own
        bins = np.full(len(lengths), np.iinfo(np.int64).min)
        is_positive = lengths > 0
        bins[is_positive] = np.round(
            np.log(lengths[is_positive]) / np.log1p(length_rtol)
        )
    else:
        bins = np.arange(len(lengths))
    episode_weights = np.asarray(weights, dtype=np.float64)[episodes]
    grouped = pd.DataFrame(
        {
            "trait": episode_traits,
            "bin": bins,
            "num_trials": episode_weights,
            "num_successes": episode_weights * gains[episodes, episode_traits],
            "length": episode_weights * lengths,
        },
    ).groupby(["trait", "bin"], sort=True)
    sums = grouped[["num_trials", "num_successes", "length"]].sum()
    rows = grouped.ngroup().to_numpy()  # each episode's merged row

    # each row's distribution of episode weight over survival durations;
    # l1 normalization scales each row to sum to 1
    survival_shares = sklearn_preprocessing.normalize(
        scipy_sparse.csr_matrix(
            (episode_weights, (rows, survival_duration_indices[episodes])),
            shape=(len(sums), num_survival_durations),
        ),
        norm="l1",
    )
    # drop rows whose episodes all have zero weight (e.g., in a bootstrap
    # replicate)
    has_episodes = sums["num_trials"].to_numpy() > 0
    partition_df = sums[has_episodes].reset_index()
    partition_df["length"] /= partition_df["num_trials"]
    partition_df["trait"] = pd.Categorical(
        partition_df["trait"], categories=range(num_traits)
    )
    return (
        partition_df[["trait", "num_trials", "num_successes", "length"]],
        survival_shares[has_episodes],
    )
