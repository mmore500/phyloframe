import typing

import numpy as np
import pandas as pd

from ._alifestd_has_contiguous_ids import alifestd_has_contiguous_ids
from ._alifestd_is_topologically_sorted import alifestd_is_topologically_sorted


def alifestd_mark_to_array(
    phylogeny_df: pd.DataFrame,
    mark_op: typing.Callable,
    mutate: bool = False,
    **kwargs: typing.Any,
) -> np.ndarray:
    """Apply a Pandas-based phyloframe mark operation, returning the marked
    values as an array rather than as a new column.

    Calls ``mark_op(phylogeny_df, mutate=True, mark_as=..., **kwargs)``
    under a temporary column name.

    Phylogeny must have contiguous ids and be topologically sorted, so that
    array positions correspond to ids; otherwise ``NotImplementedError`` is
    raised.

    Input dataframe is not mutated by this operation unless ``mutate`` set
    True.

    Parameters
    ----------
    phylogeny_df : pandas.DataFrame
        The phylogeny as a dataframe in alife standard format.
    mark_op : Callable
        A Pandas-based mark operation taking a ``mark_as`` keyword, such as
        ``alifestd_mark_lineage_cumsum_asexual``.
    mutate : bool, default False
        Are side effects on the input argument ``phylogeny_df`` allowed?
    **kwargs
        Additional keyword arguments forwarded to ``mark_op``.

    Returns
    -------
    np.ndarray
        The values marked by ``mark_op``.

    Examples
    --------
    >>> import pandas as pd
    >>> from phyloframe.legacy import (
    ...     alifestd_mark_lineage_cumsum_asexual,
    ...     alifestd_mark_to_array,
    ... )
    >>> df = pd.DataFrame(
    ...     {"id": [0, 1, 2], "ancestor_id": [0, 0, 1], "v": [1, 2, 3]},
    ... )
    >>> alifestd_mark_to_array(
    ...     df, alifestd_mark_lineage_cumsum_asexual, values="v"
    ... )
    array([1, 3, 6])

    See Also
    --------
    alifestd_mark_to_array_polars :
        Polars-based implementation.
    """
    if not alifestd_has_contiguous_ids(phylogeny_df):
        raise NotImplementedError("non-contiguous ids not supported")
    if not alifestd_is_topologically_sorted(phylogeny_df):
        raise NotImplementedError("non-topologically-sorted not supported")

    if not mutate:
        phylogeny_df = phylogeny_df.copy()

    mark_as = "_alifestd_mark_to_array"
    return mark_op(phylogeny_df, mutate=True, mark_as=mark_as, **kwargs)[
        mark_as
    ].to_numpy()
