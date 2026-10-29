import typing

import numpy as np
from scipy import spatial


def calc_kdtree_partitions(points: np.ndarray, num_levels: int) -> np.ndarray:
    """Assign points to the partitions of a balanced k-d tree.

    Recursively splitting at the median gives `2 ** num_levels` partitions
    that hold equal numbers of points, to within one, when coordinates are
    distinct (e.g., ranks, with ties broken).

    Parameters
    ----------
    points : np.ndarray
        Array of shape (num_points, num_dimensions).
    num_levels : int
        Number of median splits along each path from the root.

    Returns
    -------
    np.ndarray
        Partition of each point, as integers in `range(2 ** num_levels)`.

    Raises
    ------
    NotImplementedError
        If there are too few distinct points to split `num_levels` times.

    See Also
    --------
    scipy.spatial.cKDTree :
        Builds the tree.
    """
    points = np.asarray(points, dtype=np.float64)
    partitions = np.zeros(len(points), dtype=np.int64)
    if num_levels == 0:
        return partitions

    # tree leaves no larger than the target partitions, so splits reach
    # num_levels
    leafsize = max(1, len(points) >> (num_levels + 1))
    tree = spatial.cKDTree(points, leafsize=leafsize, balanced_tree=True)

    def assign(node: typing.Any, level: int, partition: int) -> None:
        if level == num_levels:
            partitions[node.indices] = partition
        elif node.lesser is None:
            raise NotImplementedError(
                f"too few distinct points to split {num_levels} times",
            )
        else:  # number partitions by their path of lesser/greater splits
            assign(node.lesser, level + 1, 2 * partition)
            assign(node.greater, level + 1, 2 * partition + 1)

    assign(tree.tree, 0, 0)
    return partitions
