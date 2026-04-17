import polars as pl

from ._alifestd_make_empty_polars import alifestd_make_empty_polars
from ._alifestd_make_star import _make_star_fast_path


def alifestd_make_star_polars(n_leaves: int) -> pl.DataFrame:
    r"""Build a star tree with `n_leaves` leaves.

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

    ids, ancestor_ids = _make_star_fast_path(n_leaves)
    return pl.DataFrame({"id": ids, "ancestor_id": ancestor_ids})
