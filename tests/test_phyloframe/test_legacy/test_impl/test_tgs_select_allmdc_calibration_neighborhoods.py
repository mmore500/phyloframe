import numpy as np

from phyloframe.legacy._impl import (
    tgs_count_traits,
    tgs_find_trait_exposures,
    tgs_partition_size_partitions,
    tgs_select_allmdc_calibration_neighborhoods,
)

from ._simulate_allmdc_exposure_df import simulate_allmdc_exposure_df


def test_block_diagonal():
    exposure_df = simulate_allmdc_exposure_df(4_000)
    size_partitions = (exposure_df["clade_size"] > 1).to_numpy().astype(int)
    partitions, blocks = tgs_partition_size_partitions(
        size_partitions=size_partitions,
        branch_lengths=exposure_df["branch_length"].to_numpy(),
        origin_times=exposure_df["origin_time"].to_numpy(),
        max_partitions=32,
    )
    trait_units, trait_gains = tgs_find_trait_exposures(exposure_df)
    res = tgs_select_allmdc_calibration_neighborhoods(
        counts=tgs_count_traits(
            leaf_partitions=partitions,
            num_partitions=partitions.max() + 1,
            pendant_lengths=exposure_df["branch_length"].to_numpy(),
            leaf_weights=np.ones(len(exposure_df)),
            trait_leaves=trait_units,
            trait_carriers=trait_gains,
        ),
        blocks=blocks,
        neighborhood_fractions=(0.25, 1.0),
        neutral_traits=[1, 2],
        trait_names=["focal", "n1", "n2"],
    )
    num_partitions = partitions.max() + 1
    assert res.shape == (num_partitions, num_partitions)
    (__, first, __), __ = blocks
    dense = res.toarray()
    assert not dense[:first, first:].any()
    assert not dense[first:, :first].any()
    # each partition is in its own neighborhood
    assert (np.diag(dense) == 1).all()
