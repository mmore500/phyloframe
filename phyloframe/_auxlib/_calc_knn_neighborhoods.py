import numpy as np
from scipy import sparse
from sklearn import neighbors


def calc_knn_neighborhoods(
    points: np.ndarray, *, num_neighbors: int
) -> sparse.csr_matrix:
    """Calculate each point's neighborhood of its nearest points, itself
    included.

    Parameters
    ----------
    points : np.ndarray
        Array of shape (num_points, num_dimensions).
    num_neighbors : int
        Number of points in each neighborhood, including the point itself;
        clipped to the number of points.

    Returns
    -------
    scipy.sparse.csr_matrix
        Square membership matrix, whose row `i` marks (with 1) the
        `num_neighbors` points nearest point `i`, by Euclidean distance.

    See Also
    --------
    sklearn.neighbors.NearestNeighbors.kneighbors_graph :
        Finds the neighbors.
    """
    points = np.asarray(points, dtype=np.float64)
    num_neighbors = int(np.clip(num_neighbors, 1, len(points)))
    # query with the points explicitly, so that each point is its own
    # nearest neighbor, rather than excluded as with kneighbors_graph()
    return (
        neighbors.NearestNeighbors(n_neighbors=num_neighbors)
        .fit(points)
        .kneighbors_graph(points, mode="connectivity")
    )
