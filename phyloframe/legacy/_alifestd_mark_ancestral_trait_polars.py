import gc
import logging
import typing

import numpy as np
import polars as pl

from .._auxlib._log_memory_usage import log_memory_usage
from ._alifestd_mark_lineage_cummin_polars import (
    alifestd_mark_lineage_cummin_polars,
)


def alifestd_mark_ancestral_trait_polars(
    phylogeny_df: pl.DataFrame,
    *,
    mark_as: str = "ancestral_trait",
    trait_default: typing.Any,
    trait_mask: np.ndarray,
    trait_value: np.ndarray,
) -> pl.DataFrame:
    """Mark each taxon with the trait value of the oldest taxon along its
    lineage (inclusive) that has a defined trait value, or `trait_default`
    if no such taxon exists.

    Adds a column ``mark_as``.

    Only supports asexual phylogenies.

    Parameters
    ----------
    phylogeny_df : polars.DataFrame
        The phylogeny as a dataframe in alife standard format.

        Must represent an asexual phylogeny with contiguous,
        topologically sorted ids.
    mark_as : str, default "ancestral_trait"
        Column name for the mark.
    trait_default : scalar or numpy.ndarray
        Value for taxa with no defined trait value along their lineage,
        either a scalar or an array corresponding to rows in `phylogeny_df`.
    trait_mask : numpy.ndarray
        Boolean array marking the taxa that have a defined trait value,
        corresponding to rows in `phylogeny_df`.
    trait_value : numpy.ndarray
        Array of trait values corresponding to rows in `phylogeny_df`.

    Raises
    ------
    NotImplementedError
        If `phylogeny_df` has non-contiguous ids or is not topologically
        sorted.

    Returns
    -------
    polars.DataFrame
        The phylogeny with an added mark column.

    See Also
    --------
    alifestd_mark_lineage_cummin_polars :
        Marks minimum of values along each lineage.
    """
    n = len(trait_mask)
    ids = np.arange(n)

    logging.info(
        "- alifestd_mark_ancestral_trait_polars: finding oldest marked...",
    )
    # unmarked taxa get sentinel n; ids increase from ancestor to descendant
    oldest_col = "alifestd_mark_ancestral_trait_oldest"
    oldest = (
        alifestd_mark_lineage_cummin_polars(
            phylogeny_df,
            pl.lit(pl.Series(np.where(trait_mask, ids, n))),
            mark_as=oldest_col,
        )
        .lazy()
        .select(oldest_col)
        .collect()
        .to_series()
        .to_numpy()
    )
    gc.collect()
    log_memory_usage(logging.info)

    logging.info(
        "- alifestd_mark_ancestral_trait_polars: assigning trait values...",
    )
    ancestral_trait = np.where(
        oldest < n,
        np.asarray(trait_value)[np.minimum(oldest, n - 1)],
        np.broadcast_to(trait_default, (n,)),
    )
    del ids, oldest
    gc.collect()
    log_memory_usage(logging.info)

    return phylogeny_df.with_columns(
        pl.Series(mark_as, ancestral_trait),
    )
