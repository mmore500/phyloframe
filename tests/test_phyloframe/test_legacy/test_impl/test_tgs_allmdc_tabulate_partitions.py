import numpy as np
import pytest

from phyloframe.legacy._impl import (
    tgs_allmdc_partition_episodes,
    tgs_allmdc_tabulate_partitions,
)

from ._simulate_allmdc_exposure_df import simulate_allmdc_exposure_df


def _by_trait(data, shares):
    """Each trait's (num_trials, num_successes, length, survival_shares)."""
    return [
        (
            *(partitions[c].to_numpy() for c in _COLUMNS),
            shares[partitions.index],
        )
        for __, partitions in data.groupby("trait", observed=False)
    ]


_COLUMNS = ("num_trials", "num_successes", "length")


def _partition(exposure_df, calibrate):
    return tgs_allmdc_partition_episodes(
        exposure_df=exposure_df,
        neutral_traits=[1, 2],
        trait_names=["focal", "n1", "n2"],
        calibrate=calibrate,
        calibrate_neighborhood_fractions=(0.25, 1.0),
        calibrate_num_partitions=64,
        calibrate_max_rate_ratio=np.inf,
        calibrate_size_partition_ratio=4.0,
        calibrate_size_partition_min_neutral_gains=1,
    )


@pytest.mark.parametrize("calibrate", [False, True])
def test_data(calibrate: bool):
    exposure_df = simulate_allmdc_exposure_df(4_000)
    partitioning = _partition(exposure_df, calibrate)
    data = _by_trait(
        *tgs_allmdc_tabulate_partitions(
            exposure_df=exposure_df,
            partitioning=partitioning,
            weights=np.ones(4_000),
            neutral_traits=[1, 2],
            calibrate=calibrate,
            calibrate_max_rate_ratio=np.inf,
        )
    )
    assert len(data) == 3
    for trait, (n, k, lengths, shares) in enumerate(data):
        assert n.sum() == 4_000
        assert k.sum() == exposure_df[f"trait{trait}_gain"].sum()
        assert (lengths > 0).all()
        assert shares.shape == (
            len(n),
            len(partitioning["distinct_survival_times"]),
        )
        np.testing.assert_allclose(np.asarray(shares.sum(axis=1)).ravel(), 1.0)
    if not calibrate:
        n, __, lengths, __ = data[0]
        assert (n * lengths).sum() == pytest.approx(
            exposure_df["branch_length"].sum()
        )


def test_weights():
    exposure_df = simulate_allmdc_exposure_df(4_000)
    partitioning = _partition(exposure_df, False)
    weights = np.zeros(4_000)
    weights[:100] = 2.0
    (n, __, __, shares), *__ = _by_trait(
        *tgs_allmdc_tabulate_partitions(
            exposure_df=exposure_df,
            partitioning=partitioning,
            weights=weights,
            neutral_traits=[1, 2],
            calibrate=False,
            calibrate_max_rate_ratio=np.inf,
        )
    )
    # partitions without weighted branches are omitted
    assert n.sum() == 200
    assert (n > 0).all()
    np.testing.assert_allclose(np.asarray(shares.sum(axis=1)).ravel(), 1.0)


def test_trait_without_units():
    exposure_df = simulate_allmdc_exposure_df(1_000)
    exposure_df = exposure_df.assign(
        trait0_starts_absent=False, trait0_gain=False
    )
    (n, k, lengths, shares), *__ = _by_trait(
        *tgs_allmdc_tabulate_partitions(
            exposure_df=exposure_df,
            partitioning=_partition(exposure_df, False),
            weights=np.ones(1_000),
            neutral_traits=[1, 2],
            calibrate=False,
            calibrate_max_rate_ratio=np.inf,
        )
    )
    assert len(n) == len(k) == len(lengths) == shares.shape[0] == 0


@pytest.mark.parametrize("rate", [0.0, -0.01, -0.05])
def test_survival_shares_exact(rate: float):
    exposure_df = simulate_allmdc_exposure_df(4_000)
    partitioning = _partition(exposure_df, False)
    data, shares = tgs_allmdc_tabulate_partitions(
        exposure_df=exposure_df,
        partitioning=partitioning,
        weights=np.ones(len(exposure_df)),
        neutral_traits=[1, 2],
        calibrate=False,
        calibrate_max_rate_ratio=np.inf,
    )
    times = partitioning["distinct_survival_times"]
    discounts = shares @ np.exp(rate * times)
    # brute force: average exp(rate * T) over each trait and partition's
    # branches that start without the trait
    survival = exposure_df["survival_time"].to_numpy()
    partitions = partitioning["partitions"]
    num_partitions = partitions.max() + 1
    for trait in range(3):
        at_risk = exposure_df[f"trait{trait}_starts_absent"].to_numpy()
        sums = np.bincount(
            partitions[at_risk],
            np.exp(rate * survival[at_risk]),
            num_partitions,
        )
        counts = np.bincount(partitions[at_risk], minlength=num_partitions)
        expected = (sums / np.maximum(counts, 1))[counts > 0]
        rows = (data["trait"] == trait).to_numpy()
        np.testing.assert_allclose(discounts[rows], expected, rtol=1e-12)
