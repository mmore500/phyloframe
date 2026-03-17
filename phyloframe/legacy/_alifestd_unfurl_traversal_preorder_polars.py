import logging

import numpy as np
import polars as pl

from ._alifestd_has_contiguous_ids_polars import (
    alifestd_has_contiguous_ids_polars,
)
from ._alifestd_is_topologically_sorted_polars import (
    alifestd_is_topologically_sorted_polars,
)
from ._alifestd_mark_csr_children_asexual import (
    _alifestd_mark_csr_children_asexual_fast_path,
)
from ._alifestd_mark_csr_offsets_asexual import (
    _alifestd_mark_csr_offsets_asexual_fast_path,
)
from ._alifestd_mark_num_children_asexual import (
    _alifestd_mark_num_children_asexual_fast_path,
)
from ._alifestd_try_add_ancestor_id_col_polars import (
    alifestd_try_add_ancestor_id_col_polars,
)
from ._alifestd_unfurl_traversal_preorder_asexual import (
    _alifestd_unfurl_traversal_preorder_asexual_jit,
    _alifestd_unfurl_traversal_preorder_asexual_sibling_jit,
)


def alifestd_unfurl_traversal_preorder_polars(
    phylogeny_df: pl.DataFrame,
) -> np.ndarray:
    """List node indices in DFS preorder traversal order.

    Parameters
    ----------
    phylogeny_df : polars.DataFrame or polars.LazyFrame
        The phylogeny as a dataframe in alife standard format.

        Must represent an asexual phylogeny with contiguous ids and
        topologically sorted rows.

    Returns
    -------
    np.ndarray
        Index array giving DFS preorder traversal order.

    See Also
    --------
    alifestd_unfurl_traversal_preorder_asexual :
        Pandas-based implementation.
    """
    logging.info(
        "- alifestd_unfurl_traversal_preorder_polars:"
        " adding ancestor_id col...",
    )
    phylogeny_df = alifestd_try_add_ancestor_id_col_polars(phylogeny_df)

    if phylogeny_df.lazy().limit(1).collect().is_empty():
        return np.empty(0, dtype=np.int64)

    logging.info(
        "- alifestd_unfurl_traversal_preorder_polars:"
        " checking contiguous ids...",
    )
    if not alifestd_has_contiguous_ids_polars(phylogeny_df):
        raise NotImplementedError(
            "non-contiguous ids not yet supported",
        )

    logging.info(
        "- alifestd_unfurl_traversal_preorder_polars:"
        " checking topological sort...",
    )
    if not alifestd_is_topologically_sorted_polars(phylogeny_df):
        raise NotImplementedError(
            "topologically unsorted rows not yet supported",
        )

    logging.info(
        "- alifestd_unfurl_traversal_preorder_polars:"
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
        "- alifestd_unfurl_traversal_preorder_polars:"
        " calculating preorder traversal...",
    )
    schema_names = phylogeny_df.lazy().collect_schema().names()

    # Prefer CSR-based JIT (default path, builds CSR if needed)
    has_sibling_cols = (
        "first_child_id" in schema_names and "next_sibling_id" in schema_names
    )
    if not has_sibling_cols:
        if "num_children" not in schema_names:
            num_children = _alifestd_mark_num_children_asexual_fast_path(
                ancestor_ids,
            )
        else:
            num_children = (
                phylogeny_df.lazy()
                .select("num_children")
                .collect()
                .to_series()
                .to_numpy()
            )
        if "csr_offsets" not in schema_names:
            csr_offsets = _alifestd_mark_csr_offsets_asexual_fast_path(
                ancestor_ids,
            )
        else:
            csr_offsets = (
                phylogeny_df.lazy()
                .select("csr_offsets")
                .collect()
                .to_series()
                .to_numpy()
            )
        if "csr_children" not in schema_names:
            csr_children = _alifestd_mark_csr_children_asexual_fast_path(
                ancestor_ids,
                csr_offsets,
            )
        else:
            csr_children = (
                phylogeny_df.lazy()
                .select("csr_children")
                .collect()
                .to_series()
                .to_numpy()
            )
        return _alifestd_unfurl_traversal_preorder_asexual_jit(
            ancestor_ids,
            csr_offsets,
            csr_children,
            num_children,
        )

    # Fall back to sibling-based JIT when CSR columns are absent
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
    return _alifestd_unfurl_traversal_preorder_asexual_sibling_jit(
        ancestor_ids,
        first_child_ids,
        next_sibling_ids,
    )
