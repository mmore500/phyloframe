import numpy as np

from .._alifestd_mark_lineage_cumsum_asexual import (
    _alifestd_mark_lineage_cumsum_asexual_fast_path,
)


def tgs_calc_trait_states(
    *,
    ancestor_ids: np.ndarray,
    trait_name: str,
    trait_transitions: np.ndarray,
) -> np.ndarray:
    """Implementation detail for `alifestd_tgs_pendantmdc_estimate_polars` and
    `alifestd_tgs_allmdc_estimate_polars`. Acronym tgs stands for trait
    gain/selection.

    Takes a phylogeny with contiguous, topologically sorted ids and one
    trait's transitions (see `alifestd_mark_lineage_delta_asexual`).
    Returns each node's state, summing transitions from its root down, after
    checking that transitions are +1, -1, or 0, that none are losses (not
    yet supported), that roots lack the trait (zero root transitions), and
    that states stay 0 or 1.
    """
    trait_transitions = np.asarray(trait_transitions)
    if not (
        np.issubdtype(trait_transitions.dtype, np.integer)
        or trait_transitions.dtype == np.bool_
    ):
        raise NotImplementedError(
            f"trait transitions {trait_name} must have integer dtype, got "
            f"{trait_transitions.dtype}",
        )
    trait_transitions = trait_transitions.astype(np.int64)
    if not np.isin(trait_transitions, (-1, 0, 1)).all():
        raise ValueError(
            f"trait transitions {trait_name} must be +1, -1, or 0",
        )
    if (trait_transitions == -1).any():
        raise NotImplementedError(
            f"trait transitions {trait_name} include losses (-1), which are "
            "not yet supported",
        )
    ancestor_ids = np.asarray(ancestor_ids, dtype=np.int64)
    if trait_transitions[ancestor_ids == np.arange(len(ancestor_ids))].any():
        raise NotImplementedError(
            f"trait {trait_name} is present at a root (nonzero root "
            "transition); traits present at a root, so at every node absent "
            "loss, are uninformative, so drop them",
        )
    states = _alifestd_mark_lineage_cumsum_asexual_fast_path(
        ancestor_ids,
        trait_transitions,
        reverse=False,
    )
    if not np.isin(states, (0, 1)).all():
        raise ValueError(
            f"trait transitions {trait_name} give states other than 0 and 1, "
            "e.g., from duplicate gains along a lineage",
        )
    return states
