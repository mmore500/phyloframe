import numpy as np

from phyloframe.legacy._impl import (
    tgs_calc_allmdc_group_data,
    tgs_make_allmdc_group,
)

from ._simulate_allmdc_groups import simulate_allmdc_groups


def test_uncalibrated():
    groups = simulate_allmdc_groups((4_000, 2_000))
    data = tgs_calc_allmdc_group_data(
        groups=groups,
        unit_weights=[np.ones(len(g.partitions)) for g in groups],
        neutral_traits=[1, 2],
        calibrate=False,
        num_traits=3,
    )
    assert len(data) == 3 and all(len(d) == 2 for d in data)
    for trait in range(3):
        for group, (n, k, x) in zip(groups, data[trait]):
            assert n.sum() == len(group.partitions)
            assert k.sum() == len(group.trait_gains[trait])
            # mean lengths, weighted by counts, total the group's length
            np.testing.assert_allclose(
                (n * x).sum(), group.branch_lengths.sum()
            )


def test_weights():
    groups = simulate_allmdc_groups((4_000, 2_000))
    data = tgs_calc_allmdc_group_data(
        groups=groups,
        unit_weights=[2 * np.ones(len(g.partitions)) for g in groups],
        neutral_traits=[1, 2],
        calibrate=False,
        num_traits=3,
    )
    assert data[0][0][0].sum() == 2 * len(groups[0].partitions)


def test_calibrated_without_first_group_clock():
    rng = np.random.default_rng(1)
    group = tgs_make_allmdc_group(
        branch_lengths=rng.uniform(1, 10, 1000),
        origin_times=rng.uniform(0, 10, 1000),
        trait_units=[np.arange(1000)] * 3,
        trait_gains=[np.arange(0, 1000, 50)] * 3,
        neutral_traits=[1, 2],
        trait_names=["a", "b", "c"],
        calibrate=True,
        calibration_neighborhood_fractions=(1.0,),
        num_calibration_partitions=8,
    )
    empty_gains = group._replace(
        trait_gains=[
            np.arange(0, 1000, 50),
            np.array([], int),
            np.array([], int),
        ]
    )
    data = tgs_calc_allmdc_group_data(
        groups=[empty_gains, group],
        unit_weights=[np.ones(1000)] * 2,
        neutral_traits=[1, 2],
        calibrate=True,
        num_traits=3,
    )
    # no neutral gains among single leaves, so no clock to align to
    assert data[0] == [None, None]
