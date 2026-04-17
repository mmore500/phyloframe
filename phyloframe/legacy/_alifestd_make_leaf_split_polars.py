import polars as pl

from ._alifestd_make_empty_polars import alifestd_make_empty_polars
from ._alifestd_make_leaf_split import _make_leaf_split_fast_path


def alifestd_make_leaf_split_polars(
    n_leaves: int,
    seed: int,
) -> pl.DataFrame:
    """Build a random bifurcating tree via leaf-split (Yule) sampling.

    At each step, a uniformly chosen leaf is replaced by an internal node
    with two new leaf children. This produces samples from the Yule (pure-
    birth) distribution over rooted bifurcating tree shapes.

    Parameters
    ----------
    n_leaves : int
        Number of leaf nodes in the resulting tree.
    seed : int
        Integer seed for deterministic behavior.

    Returns
    -------
    pl.DataFrame
        Phylogeny dataframe with 'id' and 'ancestor_id' columns.
    """
    if n_leaves < 0:
        raise ValueError("n_leaves must be non-negative")
    elif n_leaves == 0:
        return alifestd_make_empty_polars(ancestor_id=True)

    ids, ancestor_ids = _make_leaf_split_fast_path(n_leaves, seed)
    return pl.DataFrame({"id": ids, "ancestor_id": ancestor_ids})
