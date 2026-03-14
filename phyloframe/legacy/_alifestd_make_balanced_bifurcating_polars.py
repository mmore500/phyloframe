import polars as pl

from ._alifestd_make_empty_polars import alifestd_make_empty_polars


def alifestd_make_balanced_bifurcating_polars(
    depth: int,
) -> pl.DataFrame:
    """Build a perfectly balanced bifurcating tree of given depth.

    Parameters
    ----------
    depth : int
        Depth of the tree, where depth=1 is a single root node.

    Returns
    -------
    pl.DataFrame
        Phylogeny dataframe with 'id' and 'ancestor_id' columns.
    """
    if depth < 0:
        raise ValueError("depth must be non-negative")
    elif depth == 0:
        return alifestd_make_empty_polars(ancestor_id=True)

    ids = [0]
    ancestor_ids = [0]
    next_id = 1
    queue = [0]
    for _ in range(depth - 1):
        next_queue = []
        for parent in queue:
            for _ in range(2):
                ids.append(next_id)
                ancestor_ids.append(parent)
                next_queue.append(next_id)
                next_id += 1
        queue = next_queue

    return pl.DataFrame({"id": ids, "ancestor_id": ancestor_ids})
