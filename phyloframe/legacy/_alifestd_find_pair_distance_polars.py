import polars as pl

from ._alifestd_find_pair_mrca_id_polars import (
    alifestd_find_pair_mrca_id_polars,
)


def alifestd_find_pair_distance_polars(
    phylogeny_df: pl.DataFrame,
    first: int,
    second: int,
    *,
    criterion: str = "origin_time",
) -> float | None:
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

    criterion_col = phylogeny_df.lazy().select(criterion).collect().to_series()

    def get_val(taxon_id: int) -> float:
        return float(criterion_col.slice(taxon_id, 1).item())

    mrca_criterion = get_val(mrca_id)
    first_criterion = get_val(first)
    second_criterion = get_val(second)
    return float(
        (first_criterion - mrca_criterion)
        + (second_criterion - mrca_criterion)
    )
