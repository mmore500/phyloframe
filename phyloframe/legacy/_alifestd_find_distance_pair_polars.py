import typing

import polars as pl

from ._alifestd_find_pair_mrca_id_polars import (
    alifestd_find_pair_mrca_id_polars,
)


def alifestd_find_distance_pair_polars(
    phylogeny_df: pl.DataFrame,
    first: int,
    second: int,
    *,
    criterion: str = "origin_time",
    is_topologically_sorted: typing.Optional[bool] = None,
    has_contiguous_ids: typing.Optional[bool] = None,
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
    criterion : str, default "origin_time"
        Column name used to measure distance between taxa and their MRCA.
    is_topologically_sorted : bool, optional
        If provided, skips the topological sort check. If None
        (default), the check is performed automatically.
    has_contiguous_ids : bool, optional
        If provided, skips the contiguous ids check. If None (default),
        the check is performed automatically.

    Returns
    -------
    float or None
        The pairwise distance between the two taxa, or None if they have
        no common ancestor.

    See Also
    --------
    alifestd_find_pair_mrca_id_polars :
        Finds the MRCA id used internally by this function.
    alifestd_find_distance_pair_asexual :
        Pandas-based implementation.
    """
    mrca_id = alifestd_find_pair_mrca_id_polars(
        phylogeny_df,
        first,
        second,
        is_topologically_sorted=is_topologically_sorted,
        has_contiguous_ids=has_contiguous_ids,
    )
    if mrca_id is None:
        return None

    df_collected = phylogeny_df.lazy().collect()
    lookup = df_collected.filter(
        pl.col("id").is_in([first, second, mrca_id])
    ).select(["id", criterion])

    def get_val(taxon_id: int) -> float:
        return float(lookup.filter(pl.col("id") == taxon_id)[criterion][0])

    mrca_criterion = get_val(mrca_id)
    first_criterion = get_val(first)
    second_criterion = get_val(second)
    return float(
        (first_criterion - mrca_criterion)
        + (second_criterion - mrca_criterion)
    )
