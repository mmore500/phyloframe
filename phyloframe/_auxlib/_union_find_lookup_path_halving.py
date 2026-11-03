import numpy as np

from ._jit import jit


@jit(nopython=True)
def union_find_lookup_path_halving(parents: np.ndarray, node: int) -> int:
    """Find root of `node` in union-find forest `parents`.

    Each root is its own parent. Halves path lengths in place, repointing
    every other visited node to its grandparent.

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

    References
    ----------
    Tarjan RE, van Leeuwen J. 1984. Worst-case analysis of set union
        algorithms. Journal of the ACM 31(2):245-281.
        https://doi.org/10.1145/62.2160
    Disjoint-set data structure, "Finding set representatives." Wikipedia.
        https://en.wikipedia.org/wiki/Disjoint-set_data_structure
    """
    while parents[node] != node:
        grandparent = parents[parents[node]]
        parents[node] = grandparent
        node = grandparent
    return node
