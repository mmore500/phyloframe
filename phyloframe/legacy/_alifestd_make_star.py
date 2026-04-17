import numpy as np
import pandas as pd

from ._alifestd_make_empty import alifestd_make_empty


def _make_star_fast_path(n_leaves: int):
    """Build id and ancestor_id arrays for a star tree."""
    if n_leaves == 1:
        ids = np.zeros(1, dtype=np.int64)
        ancestor_ids = np.zeros(1, dtype=np.int64)
        return ids, ancestor_ids

    n_nodes = n_leaves + 1
    ids = np.arange(n_nodes, dtype=np.int64)
    ancestor_ids = np.zeros(n_nodes, dtype=np.int64)
    return ids, ancestor_ids


def alifestd_make_star(n_leaves: int) -> pd.DataFrame:
    r"""Build a star tree with `n_leaves` leaves.

    Structure (e.g., n_leaves=4)::

              0
            / | \ \
           1  2  3 4

    The root (id 0) has every leaf as a direct child.

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

    ids, ancestor_ids = _make_star_fast_path(n_leaves)
    ancestor_lists = [
        "[None]" if i == a else f"[{a}]" for i, a in zip(ids, ancestor_ids)
    ]
    return pd.DataFrame({"id": ids, "ancestor_list": ancestor_lists})
