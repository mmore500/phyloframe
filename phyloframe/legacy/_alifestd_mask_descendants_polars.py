import gc
import logging

import numpy as np
import polars as pl

from .._auxlib._log_memory_usage import log_memory_usage
from ._alifestd_assign_contiguous_ids_polars import (
    alifestd_assign_contiguous_ids_polars,
)
from ._alifestd_has_contiguous_ids_polars import (
    alifestd_has_contiguous_ids_polars,
)
from ._alifestd_is_topologically_sorted_polars import (
    alifestd_is_topologically_sorted_polars,
)
from ._alifestd_mask_descendants_asexual import (
    _alifestd_mask_descendants_asexual_fast_path,
)
from ._alifestd_topological_sort_polars import (
    alifestd_topological_sort_polars,
)
from ._alifestd_try_add_ancestor_id_col_polars import (
    alifestd_try_add_ancestor_id_col_polars,
)


def alifestd_mask_descendants_polars(
    phylogeny_df: pl.DataFrame,
    *,
    ancestor_mask: np.ndarray,
) -> pl.DataFrame:
    """For given ancestor nodes, create a mask identifying those nodes and all
    descendants.

    Ancestral nodes are identified by `ancestor_mask` corresponding to rows
    in `phylogeny_df`.

    The mask is returned as a new column ``alifestd_mask_descendants_polars``
    in the output DataFrame.

    Only supports asexual phylogenies.

    Parameters
    ----------
    phylogeny_df : polars.DataFrame
        The phylogeny as a dataframe in alife standard format.

        Must represent an asexual phylogeny with contiguous,
        topologically sorted ids and an ``ancestor_id`` column.
    ancestor_mask : numpy.ndarray
        Boolean array indicating ancestor nodes to propagate from.

    Raises
    ------
    NotImplementedError
        If `phylogeny_df` has no "ancestor_id" column or if ids are
        non-contiguous or not topologically sorted.

    Returns
    -------
    polars.DataFrame
        The input DataFrame with an additional boolean column
        ``alifestd_mask_descendants_polars``.
    """
    phylogeny_df = alifestd_try_add_ancestor_id_col_polars(phylogeny_df)

    if not alifestd_has_contiguous_ids_polars(phylogeny_df):
        phylogeny_df = alifestd_assign_contiguous_ids_polars(phylogeny_df)

    if not alifestd_is_topologically_sorted_polars(phylogeny_df):
        phylogeny_df = alifestd_topological_sort_polars(phylogeny_df)
        phylogeny_df = alifestd_assign_contiguous_ids_polars(phylogeny_df)

    logging.info(
        "- alifestd_mask_descendants_polars: propagating mask...",
    )
    ancestor_ids = (
        phylogeny_df.lazy()
        .select("ancestor_id")
        .collect()
        .to_series()
        .to_numpy()
    )
    mask = _alifestd_mask_descendants_asexual_fast_path(
        ancestor_ids, ancestor_mask.copy()
    )
    del ancestor_ids
    gc.collect()
    log_memory_usage(logging.info)

    return phylogeny_df.with_columns(
        alifestd_mask_descendants_polars=mask,
    )
