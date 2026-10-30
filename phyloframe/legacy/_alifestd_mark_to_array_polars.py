import typing

import numpy as np
import polars as pl


def alifestd_mark_to_array_polars(
    phylogeny_df: typing.Union[pl.DataFrame, pl.LazyFrame],
    mark_op: typing.Callable,
    **kwargs: typing.Any,
) -> np.ndarray:
    """Apply a Polars-based phyloframe mark operation, returning the marked
    values as an array rather than as a new column.

    Calls ``mark_op(phylogeny_df, mark_as=..., **kwargs)`` under a
    temporary column name.

    Parameters
    ----------
    phylogeny_df : polars.DataFrame or polars.LazyFrame
        The phylogeny as a dataframe in alife standard format.
    mark_op : Callable
        A Polars-based mark operation taking a ``mark_as`` keyword, such as
        ``alifestd_mark_lineage_cumsum_polars``.
    **kwargs
        Additional keyword arguments forwarded to ``mark_op``.

    Returns
    -------
    np.ndarray
        The values marked by ``mark_op``.

    Examples
    --------
    >>> import polars as pl
    >>> from phyloframe.legacy import (
    ...     alifestd_mark_lineage_cumsum_polars,
    ...     alifestd_mark_to_array_polars,
    ... )
    >>> df = pl.DataFrame(
    ...     {"id": [0, 1, 2], "ancestor_id": [0, 0, 1], "v": [1, 2, 3]},
    ... )
    >>> alifestd_mark_to_array_polars(
    ...     df, alifestd_mark_lineage_cumsum_polars, values="v"
    ... )
    array([1, 3, 6])

    See Also
    --------
    alifestd_mark_to_array :
        Pandas-based implementation.
    """
    mark_as = "_alifestd_mark_to_array"
    return (
        mark_op(phylogeny_df, mark_as=mark_as, **kwargs)
        .lazy()
        .select(mark_as)
        .collect()
        .to_series()
        .to_numpy()
    )
