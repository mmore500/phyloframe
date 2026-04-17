import typing

import polars as pl

from .._auxlib._with_rng_state_context import with_rng_state_context
from ._alifestd_make_edge_split import _make_edge_split_fast_path
from ._alifestd_make_empty_polars import alifestd_make_empty_polars


def alifestd_make_edge_split_polars(
    n_leaves: int,
    seed: typing.Optional[int] = None,
) -> pl.DataFrame:
    """Build a random bifurcating tree via edge-split (PDA) sampling.

    At each step, a uniformly chosen existing edge is split by inserting
    a new internal node, with a new leaf attached as its sibling. This
    produces samples from the Proportional-to-Distinguishable-Arrangements
    (PDA) distribution over rooted bifurcating tree shapes.

    Ids are contiguous but not topologically sorted; inserted internal
    nodes may have ids greater than some of their descendants. Pass the
    result through ``alifestd_topological_sort_polars`` if topological
    id order is needed.

    Parameters
    ----------
    n_leaves : int
        Number of leaf nodes in the resulting tree.
    seed : int, optional
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

    impl = (
        with_rng_state_context(seed)(_make_edge_split_fast_path)
        if seed is not None
        else _make_edge_split_fast_path
    )

    ids, ancestor_ids = impl(n_leaves)
    return pl.DataFrame({"id": ids, "ancestor_id": ancestor_ids})
