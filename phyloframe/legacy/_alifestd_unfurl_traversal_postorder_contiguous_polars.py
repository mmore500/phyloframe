import logging

import numpy as np
import polars as pl

from ._alifestd_has_contiguous_ids_polars import (
    alifestd_has_contiguous_ids_polars,
)
from ._alifestd_is_topologically_sorted_polars import (
    alifestd_is_topologically_sorted_polars,
)
from ._alifestd_mark_num_children_asexual import (
    _alifestd_mark_num_children_asexual_fast_path,
)
from ._alifestd_try_add_ancestor_id_col_polars import (
    alifestd_try_add_ancestor_id_col_polars,
)
from ._alifestd_unfurl_traversal_postorder_contiguous_asexual import (
    _alifestd_unfurl_traversal_postorder_contiguous_asexual_jit,
)


def alifestd_unfurl_traversal_postorder_contiguous_polars(
    phylogeny_df: pl.DataFrame,
) -> np.ndarray:
    """List node indices in DFS postorder traversal order, with subtree
    contiguity.

    Parameters
    ----------
    phylogeny_df : polars.DataFrame or polars.LazyFrame
        The phylogeny as a dataframe in alife standard format.

        Must represent an asexual phylogeny with contiguous ids and
        topologically sorted rows.

    Returns
    -------
    np.ndarray
        Index array giving DFS postorder traversal order.

    See Also
    --------
    alifestd_unfurl_traversal_postorder_asexual :
        Pandas-based implementation.
    """
    logging.info(
        "- alifestd_unfurl_traversal_postorder_contiguous_polars:"
        " adding ancestor_id col...",
    )
    phylogeny_df = alifestd_try_add_ancestor_id_col_polars(phylogeny_df)

    if phylogeny_df.lazy().limit(1).collect().is_empty():
        return np.empty(0, dtype=np.int64)

    logging.info(
        "- alifestd_unfurl_traversal_postorder_contiguous_polars:"
        " checking contiguous ids...",
    )
    if not alifestd_has_contiguous_ids_polars(phylogeny_df):
        raise NotImplementedError(
            "non-contiguous ids not yet supported",
        )

    logging.info(
        "- alifestd_unfurl_traversal_postorder_contiguous_polars:"
        " checking topological sort...",
    )
    if not alifestd_is_topologically_sorted_polars(phylogeny_df):
        raise NotImplementedError(
            "topologically unsorted rows not yet supported",
        )

    logging.info(
        "- alifestd_unfurl_traversal_postorder_contiguous_polars:"
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
        "- alifestd_unfurl_traversal_postorder_contiguous_polars:"
        " calculating postorder traversal...",
    )
    num_children = _alifestd_mark_num_children_asexual_fast_path(ancestor_ids)
    return _alifestd_unfurl_traversal_postorder_contiguous_asexual_jit(
        ancestor_ids,
        num_children,
    )
