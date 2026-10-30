import typing

import numpy as np
import pandas as pd


def alifestd_mark_to_array(
    phylogeny_df: pd.DataFrame,
    mark_op: typing.Callable,
    *args: typing.Any,
    **kwargs: typing.Any,
) -> np.ndarray:
    """Apply a Pandas-based phyloframe mark operation, returning the marked
    values as an array rather than as a new column.

    Calls ``mark_op(phylogeny_df, *args, mark_as=..., **kwargs)`` under a
    temporary column name. Input dataframe is not mutated.

    Parameters
    ----------
    phylogeny_df : pandas.DataFrame
        The phylogeny as a dataframe in alife standard format.
    mark_op : Callable
        A Pandas-based mark operation taking a ``mark_as`` keyword, such as
        ``alifestd_mark_lineage_cumsum_asexual``.
    *args, **kwargs
        Additional arguments forwarded to ``mark_op``.

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
    >>> alifestd_mark_to_array(df, alifestd_mark_lineage_cumsum_asexual, "v")
    array([1, 3, 6])

    See Also
    --------
    alifestd_mark_to_array_polars :
        Polars-based implementation.
    """
    mark_as = "_alifestd_mark_to_array"
    return mark_op(phylogeny_df, *args, mark_as=mark_as, **kwargs)[
        mark_as
    ].to_numpy()
