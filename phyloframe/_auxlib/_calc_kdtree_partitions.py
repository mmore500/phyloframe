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
        If there are fewer than `2 ** num_levels` points, or too few
        distinct points to split `num_levels` times.

    See Also
    --------
    scipy.spatial.cKDTree :
        Builds the tree.
    """
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 2**num_levels:
        raise NotImplementedError(
            f"splitting {num_levels} times requires at least "
            f"{2 ** num_levels} points, got {len(points)}",
        )

    # cKDTree stops splitting nodes of at most `leafsize` points; at half the
    # target partition size, every node above `num_levels` gets split
    leafsize = max(1, len(points) >> (num_levels + 1))
    tree = spatial.cKDTree(points, leafsize=leafsize, balanced_tree=True)
    partitions = np.full(len(points), -1, dtype=np.int64)

    def assign(node: typing.Any, level: int, partition: int) -> None:
        if level == num_levels:
            partitions[node.indices] = partition
        elif node.lesser is None:
            raise NotImplementedError(
                f"splitting {num_levels} times requires at least "
                f"{2 ** num_levels} distinct points",
            )
        else:  # number partitions by their path of lesser/greater splits
            assign(node.lesser, level + 1, 2 * partition)
            assign(node.greater, level + 1, 2 * partition + 1)

    assign(tree.tree, 0, 0)
    assert (partitions >= 0).all()
    return partitions
