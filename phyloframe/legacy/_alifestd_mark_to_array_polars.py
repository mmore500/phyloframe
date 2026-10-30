import typing

import numpy as np
import polars as pl

from ._alifestd_has_contiguous_ids_polars import (
    alifestd_has_contiguous_ids_polars,
)
from ._alifestd_is_topologically_sorted_polars import (
    alifestd_is_topologically_sorted_polars,
)


def alifestd_mark_to_array_polars(
    phylogeny_df: typing.Union[pl.DataFrame, pl.LazyFrame],
    mark_op: typing.Callable,
    **kwargs: typing.Any,
) -> np.ndarray:
    """Apply a Polars-based phyloframe mark operation, returning the marked
    values as an array rather than as a new column.

    Calls ``mark_op(phylogeny_df, mark_as=..., **kwargs)`` under a
    temporary column name.

    Phylogeny must have contiguous ids and be topologically sorted, so that
    array positions correspond to ids; otherwise ``NotImplementedError`` is
    raised.

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
    if not alifestd_has_contiguous_ids_polars(phylogeny_df):
        raise NotImplementedError("non-contiguous ids not supported")
    if not alifestd_is_topologically_sorted_polars(phylogeny_df):
        raise NotImplementedError("non-topologically-sorted not supported")

    mark_as = "_alifestd_mark_to_array"
    phylogeny_df = mark_op(phylogeny_df, mark_as=mark_as, **kwargs)
    # marked values must remain aligned with ids
    assert alifestd_has_contiguous_ids_polars(phylogeny_df)
    return phylogeny_df.lazy().select(mark_as).collect().to_series().to_numpy()
