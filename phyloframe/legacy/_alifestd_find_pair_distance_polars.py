import typing

import polars as pl

from .._auxlib._resolve_polars_expr import _resolve_polars_expr
from ._alifestd_find_pair_mrca_id_polars import (
    alifestd_find_pair_mrca_id_polars,
)


@_resolve_polars_expr("criterion")
def alifestd_find_pair_distance_polars(
    phylogeny_df: pl.DataFrame,
    first: int,
    second: int,
    *,
    criterion: typing.Union[str, pl.Expr] = "origin_time",
) -> typing.Optional[float]:
    """Find the pairwise distance between two taxa via their MRCA.

    The distance is computed as the sum of criterion differences between each
    taxon and their Most Recent Common Ancestor (MRCA):

        distance = (criterion[first] - criterion[mrca])
                 + (criterion[second] - criterion[mrca])

    Parameters
    ----------
    phylogeny_df : polars.DataFrame or polars.LazyFrame
        The phylogeny as a dataframe in alife standard format.

        Must represent an asexual phylogeny with contiguous ids and
        topologically sorted rows.
    first : int
        First taxon id.
    second : int
        Second taxon id.
    criterion : str or polars.Expr, default "origin_time"
        Column name or polars expression used to measure distance
        between taxa and their MRCA.

    Returns
    -------
    float or None
        The pairwise distance between the two taxa, or None if they have
        no common ancestor.

    See Also
    --------
    alifestd_find_pair_mrca_id_polars :
        Finds the MRCA id used internally by this function.
    alifestd_find_pair_distance_asexual :
        Pandas-based implementation.
    """
    mrca_id = alifestd_find_pair_mrca_id_polars(
        phylogeny_df,
        first,
        second,
    )
    if mrca_id is None:
        return None

    vals = (
        phylogeny_df.lazy()
        .select(
            pl.col(criterion).slice(first, 1).alias("first"),
            pl.col(criterion).slice(second, 1).alias("second"),
            pl.col(criterion).slice(mrca_id, 1).alias("mrca"),
        )
        .collect()
    )
    return float(
        (vals["first"].item() - vals["mrca"].item())
        + (vals["second"].item() - vals["mrca"].item())
    )
