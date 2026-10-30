import typing

import numpy as np

from .._alifestd_mask_trait_singletons_asexual import (
    _alifestd_mask_trait_singletons_asexual_fast_path,
)


def tgs_find_trait_leaves(
    *,
    ancestor_ids: np.ndarray,
    trait_mask: np.ndarray,
    is_informative: np.ndarray,
) -> typing.Tuple[np.ndarray, np.ndarray]:
    """Implementation detail for
    `alifestd_estimate_trait_gain_selection_pendant_asexual`.

    Here, tgs stands for trait gain/selection.

    Informative leaves (`is_informative`) are those used for fitting:
    leaves with a positive pendant branch length that originate at or after
    `min_leaf_origin_time`. Returns positions, among informative leaves, of
    leaves whose pendant branch started without the trait (i.e., lacking it,
    or having gained it on their pendant branch), and of those among them
    that gained the trait on their pendant branch.
    """
    trait_mask = np.asarray(trait_mask, dtype=bool)
    is_pendant_gain = _alifestd_mask_trait_singletons_asexual_fast_path(
        ancestor_ids, trait_mask
    )[is_informative]
    # leaves without the trait could still gain it on their pendant branch
    is_trait_leaf = is_pendant_gain | ~trait_mask[is_informative]
    return np.flatnonzero(is_trait_leaf), np.flatnonzero(is_pendant_gain)
