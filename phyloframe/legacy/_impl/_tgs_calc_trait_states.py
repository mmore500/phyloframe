import numpy as np

from .._alifestd_mark_lineage_cumsum_asexual import (
    _alifestd_mark_lineage_cumsum_asexual_fast_path,
)


def tgs_calc_trait_states(
    *,
    ancestor_ids: np.ndarray,
    transitions: np.ndarray,
    trait_name: str,
) -> np.ndarray:
    """Implementation detail for `alifestd_tgs_pendantmdc_estimate_polars` and
    `alifestd_tgs_allmdc_estimate_polars`. Acronym tgs stands for trait
    gain/selection.

    Takes a phylogeny with contiguous, topologically sorted ids and one
    trait's transitions (see `alifestd_mark_lineage_delta_asexual`).
    Returns each node's state, summing transitions from its root down, after
    checking that transitions are +1, -1, or 0, that none are losses (not
    yet supported), and that states stay 0 or 1.
    """
    transitions = np.asarray(transitions)
    if not (
        np.issubdtype(transitions.dtype, np.integer)
        or transitions.dtype == np.bool_
    ):
        raise NotImplementedError(
            f"trait transitions {trait_name} must have integer dtype, got "
            f"{transitions.dtype}",
        )
    transitions = transitions.astype(np.int64)
    if not np.isin(transitions, (-1, 0, 1)).all():
        raise ValueError(
            f"trait transitions {trait_name} must be +1, -1, or 0",
        )
    if (transitions == -1).any():
        raise NotImplementedError(
            f"trait transitions {trait_name} include losses (-1), which are "
            "not yet supported",
        )
    states = _alifestd_mark_lineage_cumsum_asexual_fast_path(
        np.asarray(ancestor_ids, dtype=np.int64), transitions, False
    )
    if not np.isin(states, (0, 1)).all():
        raise ValueError(
            f"trait transitions {trait_name} give states other than 0 and 1, "
            "e.g., from duplicate gains along a lineage",
        )
    return states
