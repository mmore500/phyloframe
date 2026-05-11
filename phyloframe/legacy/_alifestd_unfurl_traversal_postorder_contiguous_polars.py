import logging
import typing

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
from ._alifestd_mark_num_descendants_asexual import (
    _alifestd_mark_num_descendants_asexual_fast_path,
)
from ._alifestd_try_add_ancestor_id_col_polars import (
    alifestd_try_add_ancestor_id_col_polars,
)
from ._alifestd_unfurl_traversal_postorder_contiguous_asexual import (
    _alifestd_unfurl_traversal_postorder_contiguous_asexual_asc_jit,
    _alifestd_unfurl_traversal_postorder_contiguous_asexual_jit,
    _alifestd_unfurl_traversal_postorder_contiguous_asexual_sibling_jit,
)


def alifestd_unfurl_traversal_postorder_contiguous_polars(
    phylogeny_df: pl.DataFrame,
    child_order: typing.Optional[typing.Literal["asc", "desc"]] = None,
) -> np.ndarray:
    """List node indices in DFS postorder traversal order, with subtree
    contiguity.

    Parameters
    ----------
    phylogeny_df : polars.DataFrame or polars.LazyFrame
        The phylogeny as a dataframe in alife standard format.

        Must represent an asexual phylogeny with contiguous ids and
        topologically sorted rows.
    child_order : {"asc", "desc", None}, default None
        Order in which siblings are visited when descending the tree.
        ``"asc"`` visits smallest-id child first, ``"desc"`` visits
        largest-id child first, and ``None`` uses an arbitrary
        (implementation-defined) order.

    Returns
    -------
    np.ndarray
        Index array giving DFS postorder traversal order.

    See Also
    --------
    alifestd_unfurl_traversal_postorder_asexual :
        Pandas-based implementation.
    """
    if child_order not in (None, "asc", "desc"):
        raise ValueError(
            f"child_order must be 'asc', 'desc', or None; got {child_order!r}",
        )
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
    schema_names = phylogeny_df.lazy().collect_schema().names()

    if child_order != "desc":
        if "num_descendants" in schema_names:
            num_descendants = (
                phylogeny_df.lazy()
                .select("num_descendants")
                .collect()
                .to_series()
                .to_numpy()
            )
        else:
            num_descendants = (
                _alifestd_mark_num_descendants_asexual_fast_path(ancestor_ids)
            )
        return _alifestd_unfurl_traversal_postorder_contiguous_asexual_asc_jit(
            ancestor_ids,
            num_descendants,
        )

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
        return _alifestd_unfurl_traversal_postorder_contiguous_asexual_sibling_jit(
            ancestor_ids,
            first_child_ids,
            next_sibling_ids,
        )

    # Fall back to CSR-based JIT
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
    return _alifestd_unfurl_traversal_postorder_contiguous_asexual_jit(
        ancestor_ids,
        csr_offsets,
        csr_children,
        num_children,
    )
