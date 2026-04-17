import typing

import numpy as np
import pandas as pd

from .._auxlib._jit import jit
from .._auxlib._with_rng_state_context import with_rng_state_context
from ._alifestd_make_empty import alifestd_make_empty


@jit(nopython=True)
def _make_edge_split_fast_path(n_leaves: int):
    """Build id and ancestor_id arrays for an edge-split (PDA) tree."""
    n_nodes = 2 * n_leaves - 1
    ids = np.arange(n_nodes, dtype=np.int64)
    ancestor_ids = np.empty(n_nodes, dtype=np.int64)
    ancestor_ids[0] = 0
    if n_leaves == 1:
        return ids, ancestor_ids

    ancestor_ids[1] = 0
    ancestor_ids[2] = 0

    next_id = 3
    for _ in range(n_leaves - 2):
        victim = np.random.randint(1, next_id)
        new_internal = next_id
        new_leaf = next_id + 1
        next_id += 2
        ancestor_ids[new_internal] = ancestor_ids[victim]
        ancestor_ids[victim] = new_internal
        ancestor_ids[new_leaf] = new_internal

    return ids, ancestor_ids


def alifestd_make_edge_split(
    n_leaves: int,
    seed: typing.Optional[int] = None,
) -> pd.DataFrame:
    """Build a random bifurcating tree via edge-split (PDA) sampling.

    At each step, a uniformly chosen existing edge is split by inserting
    a new internal node, with a new leaf attached as its sibling. This
    produces samples from the Proportional-to-Distinguishable-Arrangements
    (PDA) distribution over rooted bifurcating tree shapes.

    Parameters
    ----------
    n_leaves : int
        Number of leaf nodes in the resulting tree.
    seed : int, optional
        Integer seed for deterministic behavior.

    Returns
    -------
    pd.DataFrame
        Alife-standard phylogeny dataframe with 'id' and 'ancestor_list'
        columns.

    Raises
    ------
    ValueError
        If n_leaves is negative.
    """
    if n_leaves < 0:
        raise ValueError("n_leaves must be non-negative")
    elif n_leaves == 0:
        return alifestd_make_empty()

    impl = (
        with_rng_state_context(seed)(_make_edge_split_fast_path)
        if seed is not None
        else _make_edge_split_fast_path
    )

    ids, ancestor_ids = impl(n_leaves)
    ancestor_lists = [
        "[None]" if i == a else f"[{a}]" for i, a in zip(ids, ancestor_ids)
    ]
    return pd.DataFrame({"id": ids, "ancestor_list": ancestor_lists})
