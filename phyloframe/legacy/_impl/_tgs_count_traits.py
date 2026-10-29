import typing

import numpy as np

from ._TgsCounts import TgsCounts


def tgs_count_traits(
    *,
    leaf_partitions: np.ndarray,
    num_partitions: int,
    pendant_lengths: np.ndarray,
    leaf_weights: np.ndarray,
    trait_leaves: typing.Sequence[np.ndarray],
    trait_carriers: typing.Sequence[np.ndarray],
) -> TgsCounts:
    """Implementation detail for
    `alifestd_estimate_trait_gain_selection_pendant_asexual`.

    Here, tgs stands for trait gain/selection.

    Returns, per trait, (weighted) counts of leaves and of trait carriers,
    and summed pendant length, in each partition.
    """
    shape = (len(trait_leaves), num_partitions)

    def count(leaf_sets: typing.Sequence[np.ndarray], weights: np.ndarray):
        return np.array(
            [
                np.bincount(
                    leaf_partitions[leaves],
                    weights[leaves],
                    minlength=num_partitions,
                )
                for leaves in leaf_sets
            ],
        ).reshape(shape)

    return TgsCounts(
        num_leaves=count(trait_leaves, leaf_weights),
        num_carriers=count(trait_carriers, leaf_weights),
        pendant_length_sums=count(
            trait_leaves, leaf_weights * pendant_lengths
        ),
    )
