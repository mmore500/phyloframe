import typing

import numpy as np

from .._alifestd_mark_leaves import _alifestd_mark_leaves_asexual_fast_path
from .._alifestd_mask_monomorphic_clades_asexual import (
    _alifestd_mask_monomorphic_clades_asexual_fast_path,
)


def tgs_find_trait_mdcs(
    *,
    ancestor_ids: np.ndarray,
    trait_mask: np.ndarray,
    is_informative: np.ndarray,
) -> typing.Tuple[np.ndarray, np.ndarray]:
    """Implementation detail for
    `alifestd_estimate_trait_gain_selection_allmdc_asexual`.

    Here, tgs stands for trait gain/selection.

    Takes a phylogeny with contiguous, topologically sorted ids and no
    unifurcations. Informative nodes (`is_informative`) are non-root nodes
    whose branch, from their parent, is used for fitting. Returns positions,
    among informative nodes, of those whose branch started without the
    trait (i.e., whose parent's clade does not have the trait at every
    leaf), and of those among them whose branch gained it (i.e., whose own
    clade has the trait at every leaf, so that, under parsimony without
    loss, the trait arose on their branch).

    For leaves, these are the leaves and pendant gains of
    `tgs_find_trait_leaves`.
    """
    trait_mask = np.asarray(trait_mask, dtype=bool)
    is_leaf = _alifestd_mark_leaves_asexual_fast_path(ancestor_ids)
    # propagates a representative leaf value up to each inner node, so that
    # a monomorphic clade's representative value is its leaves' shared value
    representative = is_leaf & trait_mask
    is_monomorphic = _alifestd_mask_monomorphic_clades_asexual_fast_path(
        ancestor_ids, is_leaf.copy(), representative
    )
    is_all_carrier = is_monomorphic & representative
    is_eligible = ~is_all_carrier[ancestor_ids][is_informative]
    is_gain = is_eligible & is_all_carrier[is_informative]
    return np.flatnonzero(is_eligible), np.flatnonzero(is_gain)
