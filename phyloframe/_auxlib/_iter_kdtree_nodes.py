import typing

from scipy import spatial


def iter_kdtree_nodes(
    tree: spatial.cKDTree,
) -> typing.Iterator[typing.Any]:
    """Iterate over the nodes of a k-d tree, depth first in preorder.

    Visits each node before its children, and its `lesser` subtree before
    its `greater` subtree. So, leaves come in order of the points they hold
    in `tree.indices`. Leaves are nodes whose `lesser` is None.

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
