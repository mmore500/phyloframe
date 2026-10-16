import typing

import pandas as pd

from ._alifestd_aggregate_phylogenies import alifestd_aggregate_phylogenies


def alifestd_aggregate_phylogeny_partitions(
    phylogeny_df: pd.DataFrame,
    partitioned_by: typing.Union[str, typing.Sequence[str]],
    mutate: bool = False,
) -> pd.DataFrame:
    """Split a dataframe into partitions by the values of one or more columns,
    then concatenate the partitions as independent phylogenies, reassigning
    organism ids to prevent collisions.

    Partitions are ordered by first appearance of their key in
    `phylogeny_df`, and rows retain their relative order within each
    partition. Each partition is expected to be a self-contained phylogeny,
    i.e., no organism should have an ancestor in a different partition.

    Input dataframe is not mutated by this operation unless `mutate` set True.
    If mutate set True, operation does not occur in place; still use return
    value to get transformed phylogeny dataframe.

    Parameters
    ----------
    phylogeny_df : pandas.DataFrame
        The phylogeny as a dataframe in alife standard format.
    partitioned_by : str or sequence of str
        Name(s) of the column(s) to split the dataframe over.
    mutate : bool, default False
        If True, allow mutation of the input dataframe.

    Returns
    -------
    pandas.DataFrame
        The aggregated phylogenies, with unique organism ids.

    See Also
    --------
    alifestd_aggregate_phylogeny_partitions_polars :
        Polars-based implementation.
    alifestd_aggregate_phylogenies :
        Aggregates a list of separate phylogeny dataframes.
    """
    if isinstance(partitioned_by, str):
        partitioned_by = [partitioned_by]
    else:
        partitioned_by = list(partitioned_by)
    if not partitioned_by:
        raise ValueError("partitioned_by must specify at least one column")

    missing = [col for col in partitioned_by if col not in phylogeny_df]
    if missing:
        raise ValueError(
            f"partitioned_by columns {missing!r} not found in phylogeny_df",
        )

    if len(phylogeny_df) == 0:
        return phylogeny_df if mutate else phylogeny_df.copy()

    partitions = [
        partition
        for __, partition in phylogeny_df.groupby(
            partitioned_by,
            sort=False,
            dropna=False,
        )
    ]
    # groupby slices are copies, so aggregation may safely mutate them
    return alifestd_aggregate_phylogenies(partitions, mutate=True)
