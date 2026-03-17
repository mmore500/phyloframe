import logging

import numpy as np
import polars as pl

from ._alifestd_has_contiguous_ids_polars import (
    alifestd_has_contiguous_ids_polars,
)
from ._alifestd_is_topologically_sorted_polars import (
    alifestd_is_topologically_sorted_polars,
)
from ._alifestd_mark_node_depth_asexual import (
    _alifestd_calc_node_depth_asexual_contiguous,
)
from ._alifestd_try_add_ancestor_id_col_polars import (
    alifestd_try_add_ancestor_id_col_polars,
)
from ._alifestd_unfurl_traversal_postorder_asexual import (
    _alifestd_unfurl_traversal_postorder_asexual_fast_path,
    _alifestd_unfurl_traversal_postorder_asexual_sibling_jit,
)
from ._alifestd_unfurl_traversal_postorder_contiguous_asexual import (
    _alifestd_unfurl_traversal_postorder_contiguous_asexual_jit,
)


def alifestd_unfurl_traversal_postorder_polars(
    phylogeny_df: pl.DataFrame,
) -> np.ndarray:
    """List node indices in postorder traversal order.

    Parameters
    ----------
    phylogeny_df : polars.DataFrame or polars.LazyFrame
        The phylogeny as a dataframe in alife standard format.

        Must represent an asexual phylogeny with contiguous ids and
        topologically sorted rows.

    Returns
    -------
    np.ndarray
        Index array giving postorder traversal order.

    See Also
    --------
    alifestd_unfurl_traversal_postorder_asexual :
        Pandas-based implementation.
    """
    logging.info(
        "- alifestd_unfurl_traversal_postorder_polars:"
        " adding ancestor_id col...",
    )
    phylogeny_df = alifestd_try_add_ancestor_id_col_polars(phylogeny_df)

    if phylogeny_df.lazy().limit(1).collect().is_empty():
        return np.empty(0, dtype=np.int64)

    logging.info(
        "- alifestd_unfurl_traversal_postorder_polars:"
        " checking contiguous ids...",
    )
    if not alifestd_has_contiguous_ids_polars(phylogeny_df):
        raise NotImplementedError(
            "non-contiguous ids not yet supported",
        )

    logging.info(
        "- alifestd_unfurl_traversal_postorder_polars:"
        " checking topological sort...",
    )
    if not alifestd_is_topologically_sorted_polars(phylogeny_df):
        raise NotImplementedError(
            "topologically unsorted rows not yet supported",
        )

    logging.info(
        "- alifestd_unfurl_traversal_postorder_polars:"
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
        "- alifestd_unfurl_traversal_postorder_polars:"
        " calculating postorder traversal...",
    )
    schema_names = phylogeny_df.lazy().collect_schema().names()

    # Prefer sibling-based JIT when first_child_id/next_sibling_id available
    if "first_child_id" in schema_names and "next_sibling_id" in schema_names:
        first_child_ids = (
            phylogeny_df.lazy()
            .select("first_child_id")
            .collect()
            .to_series()
            .to_numpy()
        )
        next_sibling_ids = (
            phylogeny_df.lazy()
            .select("next_sibling_id")
            .collect()
            .to_series()
            .to_numpy()
        )
        return _alifestd_unfurl_traversal_postorder_asexual_sibling_jit(
            ancestor_ids,
            first_child_ids,
            next_sibling_ids,
        )

    # Fall back to CSR-based JIT when CSR columns are present
    if (
        "csr_offsets" in schema_names
        and "csr_children" in schema_names
        and "num_children" in schema_names
    ):
        csr_offsets = (
            phylogeny_df.lazy()
            .select("csr_offsets")
            .collect()
            .to_series()
            .to_numpy()
        )
        csr_children = (
            phylogeny_df.lazy()
            .select("csr_children")
            .collect()
            .to_series()
            .to_numpy()
        )
        num_children = (
            phylogeny_df.lazy()
            .select("num_children")
            .collect()
            .to_series()
            .to_numpy()
        )
        return _alifestd_unfurl_traversal_postorder_contiguous_asexual_jit(
            ancestor_ids,
            csr_offsets,
            csr_children,
            num_children,
        )

    # Fall back to lexsort on node_depth
    if "node_depth" not in schema_names:
        node_depths = _alifestd_calc_node_depth_asexual_contiguous(
            ancestor_ids,
        )
    else:
        node_depths = (
            phylogeny_df.lazy()
            .select("node_depth")
            .collect()
            .to_series()
            .to_numpy()
        )
    return _alifestd_unfurl_traversal_postorder_asexual_fast_path(
        ancestor_ids,
        node_depths,
    )
