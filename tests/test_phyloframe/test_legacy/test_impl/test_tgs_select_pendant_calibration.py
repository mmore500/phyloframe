import logging

import numpy as np

from phyloframe.legacy._impl import (
    tgs_calc_rank_partitions,
    tgs_count_traits,
    tgs_select_pendant_calibration,
)


def test_select(caplog):
    rng = np.random.default_rng(1)
    num_leaves = 4_000
    lengths = rng.uniform(0.5, 60, num_leaves)
    origin_times = rng.uniform(0, 20, num_leaves)
    partitions, num_partitions, coords = tgs_calc_rank_partitions(
        coordinates=[lengths, origin_times], max_partitions=64
    )
    trait_carriers = [
        np.flatnonzero(rng.random(num_leaves) < 1e-2 * lengths)
        for __ in range(3)
    ]
    with caplog.at_level(logging.INFO):
        res = tgs_select_pendant_calibration(
            counts=tgs_count_traits(
                leaf_partitions=partitions,
                num_partitions=num_partitions,
                pendant_lengths=lengths,
                leaf_weights=np.ones(num_leaves),
                trait_leaves=[np.arange(num_leaves)] * 3,
                trait_carriers=trait_carriers,
            ),
            partition_coords=coords,
            neighborhood_fractions=(0.25, 1.0),
            neutral_traits=[0, 1, 2],
            trait_names=["a", "b", "c"],
            max_rate_ratio=np.inf,
        )
    assert res.shape == (num_partitions, num_partitions)
    assert "calibrated trait a" in caplog.text
