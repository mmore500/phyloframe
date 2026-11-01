import typing

import numpy as np

from .._alifestd_mask_trait_mdcs_asexual import (
    _alifestd_mask_trait_mdcs_asexual_clade_fast_path,
)


def tgs_find_trait_mdcs(
    *,
    ancestor_ids: np.ndarray,
    trait_mask: np.ndarray,
    is_informative: np.ndarray,
) -> typing.Tuple[np.ndarray, np.ndarray]:
    """Implementation detail for `alifestd_tgs_allmdc_estimate_asexual`.

    Here, tgs stands for trait gain/selection.

    Takes a phylogeny with contiguous, topologically sorted ids and no
    unifurcations. Informative nodes (`is_informative`) are non-root nodes
    whose branch, from their parent, is used for fitting. Returns positions,
    among informative nodes, of those whose branch started without the
    trait (i.e., whose parent's clade does not have the trait at every
    leaf), and of those among them that root a mutationally defined clade
    (MDC; see `alifestd_mask_trait_mdcs_asexual`), whose branch gained the
    trait under parsimony without loss.

    For leaves, these are the leaves and pendant gains of
    `tgs_find_trait_leaves`.
    """
    is_all_carrier = _alifestd_mask_trait_mdcs_asexual_clade_fast_path(
        ancestor_ids, trait_mask
    )
    is_eligible = ~is_all_carrier[ancestor_ids][is_informative]
    is_mdc = is_eligible & is_all_carrier[is_informative]
    return np.flatnonzero(is_eligible), np.flatnonzero(is_mdc)
