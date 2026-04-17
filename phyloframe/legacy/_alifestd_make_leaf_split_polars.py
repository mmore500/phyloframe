import typing

import polars as pl

from .._auxlib._with_rng_state_context import with_rng_state_context
from ._alifestd_make_empty_polars import alifestd_make_empty_polars
from ._alifestd_make_leaf_split import _make_leaf_split_fast_path


def alifestd_make_leaf_split_polars(
    n_leaves: int,
    seed: typing.Optional[int] = None,
) -> pl.DataFrame:
    """Build a random bifurcating tree via leaf-split (Yule) sampling.

    At each step, a uniformly chosen leaf is replaced by an internal node
    with two new leaf children. This produces samples from the Yule (pure-
    birth) distribution over rooted bifurcating tree shapes.

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
        with_rng_state_context(seed)(_make_leaf_split_fast_path)
        if seed is not None
        else _make_leaf_split_fast_path
    )

    ids, ancestor_ids = impl(n_leaves)
    return pl.DataFrame({"id": ids, "ancestor_id": ancestor_ids})
