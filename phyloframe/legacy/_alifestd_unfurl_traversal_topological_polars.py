import logging

import numpy as np
import polars as pl

from ._alifestd_has_contiguous_ids_polars import (
    alifestd_has_contiguous_ids_polars,
)
from ._alifestd_is_topologically_sorted_polars import (
    alifestd_is_topologically_sorted_polars,
)
from ._alifestd_topological_sort import _topological_sort_fast_path
from ._alifestd_try_add_ancestor_id_col_polars import (
    alifestd_try_add_ancestor_id_col_polars,
)


def alifestd_unfurl_traversal_topological_polars(
    phylogeny_df: pl.DataFrame,
) -> np.ndarray:
    """List node indices in topological traversal order.

    Parents are visited before children. If the dataframe is already
    topologically sorted, the existing row indices are returned directly.
    Otherwise, a topological ordering is computed.

    Parameters
    ----------
    phylogeny_df : polars.DataFrame or polars.LazyFrame
        The phylogeny as a dataframe in alife standard format.

        Must represent an asexual phylogeny with contiguous ids.

    Returns
    -------
    np.ndarray
        Index array giving topological traversal order.

    See Also
    --------
    alifestd_unfurl_traversal_topological_asexual :
        Pandas-based implementation.
    """
    logging.info(
        "- alifestd_unfurl_traversal_topological_polars:"
        " adding ancestor_id col...",
    )
    phylogeny_df = alifestd_try_add_ancestor_id_col_polars(phylogeny_df)

    if phylogeny_df.lazy().limit(1).collect().is_empty():
        return np.empty(0, dtype=np.int64)

    logging.info(
        "- alifestd_unfurl_traversal_topological_polars:"
        " checking contiguous ids...",
    )
    if not alifestd_has_contiguous_ids_polars(phylogeny_df):
        raise NotImplementedError(
            "non-contiguous ids not yet supported",
        )

    logging.info(
        "- alifestd_unfurl_traversal_topological_polars:"
        " checking topological sort...",
    )
    if alifestd_is_topologically_sorted_polars(phylogeny_df):
        n = phylogeny_df.lazy().select(pl.len()).collect().item()
        return np.arange(n, dtype=np.int64)

    logging.info(
        "- alifestd_unfurl_traversal_topological_polars:"
        " extracting ancestor ids...",
    )
    ancestor_ids = (
        phylogeny_df.lazy()
        .select("ancestor_id")
        .collect()
        .to_series()
        .to_numpy()
    )

    logging.info(
        "- alifestd_unfurl_traversal_topological_polars:"
        " computing topological order...",
    )
    return _topological_sort_fast_path(ancestor_ids)
