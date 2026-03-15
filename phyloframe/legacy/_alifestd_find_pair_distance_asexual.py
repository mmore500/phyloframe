import typing

import pandas as pd

from ._alifestd_find_pair_mrca_id_asexual import (
    alifestd_find_pair_mrca_id_asexual,
)


def alifestd_find_pair_distance_asexual(
    phylogeny_df: pd.DataFrame,
    first: int,
    second: int,
    *,
    criterion: str = "origin_time",
    mutate: bool = False,
) -> typing.Optional[float]:
    """Find the pairwise distance between two taxa via their MRCA.

    The distance is computed as the sum of criterion differences between each
    taxon and their Most Recent Common Ancestor (MRCA):

        distance = (criterion[first] - criterion[mrca])
                 + (criterion[second] - criterion[mrca])

    Parameters
    ----------
    phylogeny_df : pd.DataFrame
        Phylogeny in alife standard format.
    first : int
        First taxon id.
    second : int
        Second taxon id.
    criterion : str, default "origin_time"
        Column name used to measure distance between taxa and their MRCA.
    mutate : bool, default False
        If True, allows in-place modification of `phylogeny_df`.

    Returns
    -------
    float or None
        The pairwise distance between the two taxa, or None if they have
        no common ancestor.

    See Also
    --------
    alifestd_find_pair_mrca_id_asexual :
        Finds the MRCA id used internally by this function.
    alifestd_find_pair_distance_polars :
        Polars-based implementation.
    """
    mrca_id = alifestd_find_pair_mrca_id_asexual(
        phylogeny_df,
        first,
        second,
        mutate=mutate,
    )
    if mrca_id is None:
        return None

    indexed = phylogeny_df.set_index("id")[criterion]
    mrca_criterion = indexed[mrca_id]
    first_criterion = indexed[first]
    second_criterion = indexed[second]
    return float(
        (first_criterion - mrca_criterion)
        + (second_criterion - mrca_criterion)
    )
