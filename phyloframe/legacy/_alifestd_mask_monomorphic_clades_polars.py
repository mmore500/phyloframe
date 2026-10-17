import gc
import logging

import numpy as np
import pandas as pd
import polars as pl

from .._auxlib._log_memory_usage import log_memory_usage
from ._alifestd_has_contiguous_ids_polars import (
    alifestd_has_contiguous_ids_polars,
)
from ._alifestd_is_topologically_sorted_polars import (
    alifestd_is_topologically_sorted_polars,
)
from ._alifestd_mask_monomorphic_clades_asexual import (
    _alifestd_mask_monomorphic_clades_asexual_fast_path,
)
from ._alifestd_try_add_ancestor_id_col_polars import (
    alifestd_try_add_ancestor_id_col_polars,
)


def alifestd_mask_monomorphic_clades_polars(
    phylogeny_df: pl.DataFrame,
    *,
    mark_as: str = "alifestd_mask_monomorphic_clades_polars",
    trait_mask: np.ndarray,
    trait_values: np.ndarray,
) -> pl.DataFrame:
    """Compute a mask marking "monomorphic" clades where all members with a
    trait defined value share the same trait value.

    Clades containing no members with a defined trait value are considered
    monomorphic. All leaf nodes are considered monomorphic.

    The mask is returned as a new column
    ``alifestd_mask_monomorphic_clades_polars`` in the output DataFrame. Pass
    ``mark_as`` parameter to override.

    Only supports asexual phylogenies.

    Parameters
    ----------
    phylogeny_df : polars.DataFrame
        The phylogeny as a dataframe in alife standard format.

        Must represent an asexual phylogeny with contiguous,
        topologically sorted ids.
    mark_as : str, default "alifestd_mask_monomorphic_clades_polars"
        Column name for the mask.
    trait_mask : numpy.ndarray
        Boolean array marking the nodes that have a defined trait value,
        corresponding to rows in `phylogeny_df`.
    trait_values : numpy.ndarray
        Array of trait values corresponding to rows in `phylogeny_df`.

    Raises
    ------
    NotImplementedError
        If `phylogeny_df` has non-contiguous ids or is not topologically
        sorted.

    Returns
    -------
    polars.DataFrame
        The input DataFrame with an additional boolean column
        ``alifestd_mask_monomorphic_clades_polars``.

    See Also
    --------
    alifestd_mask_monomorphic_clades_asexual :
        Pandas-based implementation.
    """
    phylogeny_df = alifestd_try_add_ancestor_id_col_polars(phylogeny_df)

    if not alifestd_has_contiguous_ids_polars(phylogeny_df):

        raise NotImplementedError(
            "non-contiguous ids not supported",
        )

    if not alifestd_is_topologically_sorted_polars(phylogeny_df):

        raise NotImplementedError(
            "non-topologically-sorted data not supported",
        )

    # replace non-integer dtypes with integer aliases, so NaN/None match
    if not np.issubdtype(trait_values.dtype, np.integer):
        trait_values = pd.factorize(trait_values)[0]
    else:  # copy, modified in place
        trait_values = trait_values.copy()

    logging.info(
        "- alifestd_mask_monomorphic_clades_polars: computing mask...",
    )
    ancestor_ids = (
        phylogeny_df.lazy()
        .select("ancestor_id")
        .collect()
        .to_series()
        .to_numpy()
    )
    mask = _alifestd_mask_monomorphic_clades_asexual_fast_path(
        ancestor_ids,
        np.array(trait_mask, dtype=np.bool_),  # copy, modified in place
        trait_values,
    )
    del ancestor_ids
    gc.collect()
    log_memory_usage(logging.info)

    return phylogeny_df.with_columns(
        pl.Series(name=mark_as, values=mask),
    )
