import polars as pl

from ._alifestd_make_empty_polars import alifestd_make_empty_polars


def alifestd_make_comb_polars(n_leaves: int) -> pl.DataFrame:
    r"""Build a comb/caterpillar tree with `n_leaves` leaves.

    Parameters
    ----------
    n_leaves : int
        Number of leaf nodes in the resulting tree.

    Returns
    -------
    pl.DataFrame
        Phylogeny dataframe with 'id' and 'ancestor_id' columns.
    """
    if n_leaves < 0:
        raise ValueError("n_leaves must be non-negative")
    elif n_leaves == 0:
        return alifestd_make_empty_polars(ancestor_id=True)

    ids = [0]
    ancestor_ids = [0]
    parents = list(range(0, 2 * n_leaves, 2))
    for i in range(len(parents) - 1):
        parent = parents[i]
        child_internal = parents[i + 1]
        child_leaf = child_internal - 1
        ids.extend([child_leaf, child_internal])
        ancestor_ids.extend([parent, parent])

    return pl.DataFrame({"id": ids, "ancestor_id": ancestor_ids})
