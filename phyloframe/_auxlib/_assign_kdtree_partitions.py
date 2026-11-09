import numpy as np
from scipy import spatial

from ._iter_kdtree_nodes import iter_kdtree_nodes


def assign_kdtree_partitions(points: np.ndarray, leafsize: int) -> np.ndarray:
    """Assign points to the leaves of a balanced k-d tree.

    Partitions hold `leafsize / 2` to `leafsize` points, if coordinates are
    distinct.

    Parameters
    ----------
    points : np.ndarray
        Array of shape (num_points, num_dimensions).
    leafsize : int
        Largest number of points per partition.

    Returns
    -------
    np.ndarray
        Partition of each point, as integers numbered from 0 in tree order.
    """
    points = np.asarray(points, dtype=np.float64)
    if len(points) == 0:
        return np.zeros(0, dtype=np.int64)

    tree = spatial.cKDTree(points, leafsize=leafsize, balanced_tree=True)
    leaves = [node for node in iter_kdtree_nodes(tree) if node.lesser is None]
    partitions = np.empty(len(points), dtype=np.int64)
    partitions[tree.indices] = np.repeat(
        np.arange(len(leaves)), [node.children for node in leaves]
    )
    return partitions
