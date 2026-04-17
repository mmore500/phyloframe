import polars as pl

from ._alifestd_make_edge_split import _make_edge_split_fast_path
from ._alifestd_make_empty_polars import alifestd_make_empty_polars


def alifestd_make_edge_split_polars(
    n_leaves: int,
    seed: int,
) -> pl.DataFrame:
    """Build a random bifurcating tree via edge-split (PDA) sampling.

    At each step, a uniformly chosen existing edge is split by inserting
    a new internal node, with a new leaf attached as its sibling. This
    produces samples from the Proportional-to-Distinguishable-Arrangements
    (PDA) distribution over rooted bifurcating tree shapes.

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

    ids, ancestor_ids = _make_edge_split_fast_path(n_leaves, seed)
    return pl.DataFrame({"id": ids, "ancestor_id": ancestor_ids})
