import logging

import numpy as np
import polars as pl

from ._alifestd_find_leaf_ids_polars import alifestd_find_leaf_ids_polars
from ._alifestd_has_contiguous_ids_polars import (
    alifestd_has_contiguous_ids_polars,
)
from ._alifestd_has_multiple_roots_polars import (
    alifestd_has_multiple_roots_polars,
)
from ._alifestd_is_strictly_bifurcating_polars import (
    alifestd_is_strictly_bifurcating_polars,
)
from ._alifestd_is_topologically_sorted_polars import (
    alifestd_is_topologically_sorted_polars,
)
from ._alifestd_mark_num_descendants_asexual import (
    _alifestd_mark_num_descendants_asexual_fast_path,
)
from ._alifestd_try_add_ancestor_id_col_polars import (
    alifestd_try_add_ancestor_id_col_polars,
)
from ._alifestd_unfurl_traversal_semiorder_asexual import (
    _alifestd_unfurl_traversal_semiorder_asexual_fast_path,
)


def alifestd_unfurl_traversal_semiorder_polars(
    phylogeny_df: pl.DataFrame,
) -> np.ndarray:
    """List node indices in semiorder traversal order.

    An inorder traversal where either left child (smaller id) or right child
    (larger id) may be visited first.

    Parameters
    ----------
    phylogeny_df : polars.DataFrame or polars.LazyFrame
        The phylogeny as a dataframe in alife standard format.

        Must represent an asexual, strictly bifurcating phylogeny with
        contiguous ids and topologically sorted rows.

    Returns
    -------
    np.ndarray
        Index array giving semiorder traversal order.

    See Also
    --------
    alifestd_unfurl_traversal_semiorder_asexual :
        Pandas-based implementation.
    """
    logging.info(
        "- alifestd_unfurl_traversal_semiorder_polars:"
        " adding ancestor_id col...",
    )
    phylogeny_df = alifestd_try_add_ancestor_id_col_polars(phylogeny_df)

    if phylogeny_df.lazy().limit(1).collect().is_empty():
        return np.empty(0, dtype=np.int64)

    logging.info(
        "- alifestd_unfurl_traversal_semiorder_polars:"
        " checking single root...",
    )
    if alifestd_has_multiple_roots_polars(phylogeny_df):
        raise ValueError(
            "Phylogeny must have a single root for inorder traversal."
        )

    logging.info(
        "- alifestd_unfurl_traversal_semiorder_polars:"
        " checking strictly bifurcating...",
    )
    if not alifestd_is_strictly_bifurcating_polars(phylogeny_df):
        raise ValueError(
            "Phylogeny must be strictly bifurcating for inorder traversal.",
        )

    logging.info(
        "- alifestd_unfurl_traversal_semiorder_polars:"
        " checking contiguous ids...",
    )
    if not alifestd_has_contiguous_ids_polars(phylogeny_df):
        raise NotImplementedError(
            "non-contiguous ids not yet supported",
        )

    logging.info(
        "- alifestd_unfurl_traversal_semiorder_polars:"
        " checking topological sort...",
    )
    if not alifestd_is_topologically_sorted_polars(phylogeny_df):
        raise NotImplementedError(
            "topologically unsorted rows not yet supported",
        )

    logging.info(
        "- alifestd_unfurl_traversal_semiorder_polars:"
        " extracting ancestor ids...",
    )
    ancestor_ids = (
        phylogeny_df.lazy()
        .select("ancestor_id")
        .collect()
        .to_series()
        .to_numpy()
        .astype(np.intp)
    )

    schema_names = phylogeny_df.lazy().collect_schema().names()
    if "num_descendants" not in schema_names:
        logging.info(
            "- alifestd_unfurl_traversal_semiorder_polars:"
            " calculating num_descendants...",
        )
        num_descendants = _alifestd_mark_num_descendants_asexual_fast_path(
            ancestor_ids,
        )
    else:
        logging.info(
            "- alifestd_unfurl_traversal_semiorder_polars:"
            " selecting num_descendants...",
        )
        num_descendants = (
            phylogeny_df.lazy()
            .select("num_descendants")
            .collect()
            .to_series()
            .to_numpy()
            .astype(np.intp)
        )

    logging.info(
        "- alifestd_unfurl_traversal_semiorder_polars:" " finding leaf ids...",
    )
    leaf_ids = alifestd_find_leaf_ids_polars(phylogeny_df).astype(np.intp)

    logging.info(
        "- alifestd_unfurl_traversal_semiorder_polars:"
        " calculating semiorder traversal...",
    )
    return _alifestd_unfurl_traversal_semiorder_asexual_fast_path(
        ancestor_ids,
        num_descendants,
        leaf_ids,
    )
