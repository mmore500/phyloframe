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
    """List `id` values in topological traversal order for contiguous ids,
    or row indices for non-contiguous ids.

    Parents are visited before children. If the dataframe is already
    topologically sorted, the existing order is returned directly.
    Otherwise, a topological ordering is computed.

    Parameters
    ----------
    phylogeny_df : polars.DataFrame or polars.LazyFrame
        The phylogeny as a dataframe in alife standard format.

        Must represent an asexual phylogeny.

    Returns
    -------
    np.ndarray
        If ids are contiguous, array of `id` values in topological order.
        Otherwise, index array giving topological traversal order.

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
    is_contiguous = alifestd_has_contiguous_ids_polars(phylogeny_df)

    logging.info(
        "- alifestd_unfurl_traversal_topological_polars:"
        " checking topological sort...",
    )
    if alifestd_is_topologically_sorted_polars(phylogeny_df):
        if is_contiguous:
            return (
                phylogeny_df.lazy()
                .select("id")
                .collect()
                .to_series()
                .to_numpy()
            )
        else:
            n = phylogeny_df.lazy().select(pl.len()).collect().item()
            return np.arange(n, dtype=np.int64)

    if not is_contiguous:
        raise NotImplementedError(
            "non-contiguous ids that are not topologically sorted"
            " not yet supported",
        )

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
    order = _topological_sort_fast_path(ancestor_ids)
    ids = phylogeny_df.lazy().select("id").collect().to_series().to_numpy()
    return ids[order]
