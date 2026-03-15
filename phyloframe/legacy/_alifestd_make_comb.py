from .._auxlib._jit import jit
import numpy as np
import pandas as pd

from ._alifestd_make_empty import alifestd_make_empty


@jit(nopython=True)
def _make_comb_fast_path(n_leaves: int):
    """Build id and ancestor_id arrays for a comb tree."""
    n_nodes = 2 * n_leaves - 1
    ids = np.empty(n_nodes, dtype=np.int64)
    ancestor_ids = np.empty(n_nodes, dtype=np.int64)
    ids[0] = 0
    ancestor_ids[0] = 0
    idx = 1
    for i in range(n_leaves - 1):
        parent = 2 * i
        child_internal = 2 * (i + 1)
        child_leaf = child_internal - 1
        ids[idx] = child_leaf
        ancestor_ids[idx] = parent
        idx += 1
        ids[idx] = child_internal
        ancestor_ids[idx] = parent
        idx += 1
    return ids, ancestor_ids


def alifestd_make_comb(n_leaves: int) -> pd.DataFrame:
    r"""Build a comb/caterpillar tree with `n_leaves` leaves.

    Structure (e.g., n_leaves=4)::

              0
             / \
            1   2
               / \
              3   4
                 / \
                5   6

    Internal nodes: 0, 2, 4, ...
    Leaves: 1, 3, 5, ...

    Parameters
    ----------
    n_leaves : int
        Number of leaf nodes in the resulting tree.

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

    ids, ancestor_ids = _make_comb_fast_path(n_leaves)
    ancestor_lists = [
        "[None]" if i == a else f"[{a}]" for i, a in zip(ids, ancestor_ids)
    ]
    return pd.DataFrame({"id": ids, "ancestor_list": ancestor_lists})
