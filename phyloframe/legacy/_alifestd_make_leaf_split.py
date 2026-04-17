import typing

import numpy as np
import pandas as pd

from .._auxlib._jit import jit
from .._auxlib._with_rng_state_context import with_rng_state_context
from ._alifestd_make_empty import alifestd_make_empty


@jit(nopython=True)
def _make_leaf_split_fast_path(n_leaves: int):
    """Build id and ancestor_id arrays for a leaf-split (Yule) tree."""
    n_nodes = 2 * n_leaves - 1
    ids = np.arange(n_nodes, dtype=np.int64)
    ancestor_ids = np.zeros(n_nodes, dtype=np.int64)
    if n_leaves == 1:
        return ids, ancestor_ids

    leaves = np.zeros(n_leaves, dtype=np.int64)
    random_indices = (
        np.random.random(n_leaves - 1) * np.arange(1, n_leaves)
    ).astype(np.int64)

    for i, left in enumerate(range(1, n_nodes, 2)):
        right = left + 1
        idx = random_indices[i]
        parent = leaves[idx]
        ancestor_ids[left] = parent
        ancestor_ids[right] = parent
        leaves[idx] = left
        leaves[i + 1] = right

    return ids, ancestor_ids


def alifestd_make_leaf_split(
    n_leaves: int,
    seed: typing.Optional[int] = None,
) -> pd.DataFrame:
    """Build a random bifurcating tree via leaf-split (Yule) sampling.

    At each step, a uniformly chosen leaf is replaced by an internal node
    with two new leaf children. This produces samples from the Yule (pure-
    birth) distribution over rooted bifurcating tree shapes.

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
        with_rng_state_context(seed)(_make_leaf_split_fast_path)
        if seed is not None
        else _make_leaf_split_fast_path
    )

    ids, ancestor_ids = impl(n_leaves)
    ancestor_lists = [
        "[None]" if i == a else f"[{a}]" for i, a in zip(ids, ancestor_ids)
    ]
    return pd.DataFrame({"id": ids, "ancestor_list": ancestor_lists})
