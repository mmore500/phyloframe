import typing

import numpy as np
from scipy import spatial


def calc_kdtree_cells(points: np.ndarray, num_levels: int) -> np.ndarray:
    """Assign points to the cells of a balanced k-d tree partition.

    Recursively splitting at the median gives `2 ** num_levels` cells that
    hold equal numbers of points, to within one, when coordinates are
    distinct (e.g., ranks, with ties broken). Cells adapt to where points
    are, so they stay equally occupied even when coordinates are
    correlated, unlike grids from per-coordinate quantiles.

    Parameters
    ----------
    points : np.ndarray
        Array of shape (num_points, num_dimensions).
    num_levels : int
        Number of median splits along each path from the root.

    Returns
    -------
    np.ndarray
        Cell of each point, as integers in `range(2 ** num_levels)`. With
        fewer than `2 ** num_levels` points, some cells are empty.

    See Also
    --------
    scipy.spatial.cKDTree :
        Builds the partition.
    """
    points = np.asarray(points, dtype=np.float64)
    cells = np.zeros(len(points), dtype=np.int64)
    if len(points) == 0 or num_levels == 0:
        return cells

    # leaves no larger than the target cells, so splits reach num_levels
    leafsize = max(1, len(points) >> (num_levels + 1))
    tree = spatial.cKDTree(points, leafsize=leafsize, balanced_tree=True)

    def assign(node: typing.Any, level: int, cell: int) -> None:
        if level == num_levels or node.lesser is None:
            # a node too small to split covers all cells below it
            cells[node.indices] = cell << (num_levels - level)
            return
        assign(node.lesser, level + 1, 2 * cell)
        assign(node.greater, level + 1, 2 * cell + 1)

    assign(tree.tree, 0, 0)
    return cells
