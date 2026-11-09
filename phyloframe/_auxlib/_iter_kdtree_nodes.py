import typing

from scipy import spatial


def iter_kdtree_nodes(
    tree: spatial.cKDTree,
) -> typing.Iterator[typing.Any]:
    """Iterate over the nodes of a k-d tree, depth first in preorder.

    Visits `lesser` before `greater` subtrees, so leaves (nodes whose
    `lesser` is None) come in order of the points they hold in `tree.indices`.

    Parameters
    ----------
    tree : scipy.spatial.cKDTree
        Tree to iterate over.

    Yields
    ------
    cKDTreeNode
        Each node of the tree, starting at the root.
    """
    stack = [tree.tree]
    while stack:
        node = stack.pop()
        yield node
        if node.lesser is not None:
            stack.extend([node.greater, node.lesser])  # pop lesser first
