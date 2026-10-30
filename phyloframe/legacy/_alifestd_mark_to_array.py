import typing

import numpy as np
import pandas as pd


def alifestd_mark_to_array(
    mark_op: typing.Callable,
    ancestor_ids: np.ndarray,
    *values: np.ndarray,
    **kwargs: typing.Any,
) -> np.ndarray:
    """Apply a Pandas-based phyloframe mark operation to raw arrays,
    returning the marked values as an array.

    Builds a phylogeny with contiguous ids from ``ancestor_ids``, adds each
    of ``values`` as a column, then calls
    ``mark_op(phylogeny_df, *value_column_names, mark_as=..., **kwargs)``.

    Parameters
    ----------
    mark_op : Callable
        A Pandas-based mark operation, such as
        ``alifestd_mark_lineage_cumsum_asexual``, taking its value column
        names as positional arguments after the phylogeny dataframe.
    ancestor_ids : np.ndarray
        Ancestor id of each node, where ids correspond to array position.
    *values : np.ndarray
        Value arrays, one per value column parameter of ``mark_op``.
    **kwargs
        Additional keyword arguments forwarded to ``mark_op``.

    Returns
    -------
    np.ndarray
        The column marked by ``mark_op``.

    Examples
    --------
    >>> import numpy as np
    >>> from phyloframe.legacy import (
    ...     alifestd_mark_lineage_cumsum_asexual,
    ...     alifestd_mark_to_array,
    ... )
    >>> alifestd_mark_to_array(
    ...     alifestd_mark_lineage_cumsum_asexual,
    ...     np.array([0, 0, 1]),
    ...     np.array([1, 2, 3]),
    ... )
    array([1, 3, 6])

    See Also
    --------
    alifestd_mark_to_array_polars :
        Polars-based implementation.
    """
    ancestor_ids = np.asarray(ancestor_ids)
    value_names = [f"_value{i}" for i in range(len(values))]
    phylogeny_df = pd.DataFrame(
        {
            "id": np.arange(len(ancestor_ids)),
            "ancestor_id": ancestor_ids,
            **dict(zip(value_names, values)),
        },
    )
    return mark_op(
        phylogeny_df, *value_names, mutate=True, mark_as="_marked", **kwargs
    )["_marked"].to_numpy()
