import numpy as np

from ._jit import jit


@jit(nopython=True)
def union_find_lookup_path_halving(parents: np.ndarray, node: int) -> int:
    """Find root of `node` in union-find forest `parents`.

    Each root is its own parent. Halves path lengths in place, repointing
    every other visited node to its grandparent, so repeated lookups stay
    fast.

    Parameters
    ----------
    parents : np.ndarray
        Parent of each node in union-find forest. Modified in place.
    node : int
        Node to look up.

    Returns
    -------
    int
        Root of `node`'s tree.
    """
    while parents[node] != node:
        grandparent = parents[parents[node]]
        parents[node] = grandparent
        node = grandparent
    return node
