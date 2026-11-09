import numpy as np
import pytest
from scipy import spatial

from phyloframe._auxlib._iter_kdtree_nodes import iter_kdtree_nodes


@pytest.mark.parametrize("num_points", [1, 10, 1_000])
@pytest.mark.parametrize("leafsize", [1, 16])
def test_visits_every_node_in_preorder(num_points: int, leafsize: int):
    points = np.random.default_rng(1).random((num_points, 2))
    tree = spatial.cKDTree(points, leafsize=leafsize)
    nodes = list(iter_kdtree_nodes(tree))

    assert nodes[0].children == num_points  # root first
    leaves = [node for node in nodes if node.lesser is None]
    assert len(nodes) == 2 * len(leaves) - 1  # full binary tree
    # leaves, in order, tile the tree's point order
    np.testing.assert_array_equal(
        np.concatenate([node.indices for node in leaves]), tree.indices
    )
    for i, node in enumerate(nodes):
        if node.lesser is not None:  # lesser child immediately follows
            assert nodes[i + 1].start_idx == node.start_idx
            assert nodes[i + 1].end_idx == node.lesser.end_idx
