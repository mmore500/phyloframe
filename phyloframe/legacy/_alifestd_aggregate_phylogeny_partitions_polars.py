import typing

import polars as pl

_prefix = "__phyloframe_aggregate_partitions_"
_key_prefix = f"{_prefix}key_"
_first_name = f"{_prefix}first"
_part_name = f"{_prefix}part"
_row_name = f"{_prefix}row"
_span_name = f"{_prefix}span"
_offset_name = f"{_prefix}offset"


def alifestd_aggregate_phylogeny_partitions_polars(
    phylogeny_df: typing.Union[pl.DataFrame, pl.LazyFrame],
    partitioned_by: typing.Union[
        str,
        pl.Expr,
        typing.Sequence[typing.Union[str, pl.Expr]],
    ],
) -> typing.Union[pl.DataFrame, pl.LazyFrame]:
    """Split a dataframe into partitions by the values of one or more columns
    or expressions, then aggregate the partitions as independent phylogenies,
    reassigning organism ids to prevent collisions.

    Equivalent to splitting `phylogeny_df` over `partitioned_by` and calling
    ``alifestd_aggregate_phylogenies_polars``, but performed as a single
    query plan without collecting intermediate results. A LazyFrame input
    yields a LazyFrame (nothing is collected); a DataFrame input yields a
    DataFrame.

    Partitions are ordered by first appearance of their key in
    `phylogeny_df`, and rows retain their relative order within each
    partition. Each partition is expected to be a self-contained phylogeny,
    i.e., no organism should have an ancestor in a different partition.

    Ids within each partition are shifted by a fixed offset, so resulting ids
    are unique, and are contiguous if the ids within every partition are
    contiguous (i.e., each id equals its row number within its partition).

    Assumes asexual phylogenies with an ``ancestor_id`` column (not
    ``ancestor_list``). Topological sorting is not required; the result is
    topologically sorted if and only if the input is.

    Parameters
    ----------
    phylogeny_df : polars.DataFrame or polars.LazyFrame
        The phylogeny as a dataframe in alife standard format.
    partitioned_by : str, polars.Expr, or sequence of str and polars.Expr
        Column name(s) or expression(s) to split the dataframe over.

    See Also
    --------
    alifestd_aggregate_phylogeny_partitions :
        Pandas-based implementation.
    alifestd_aggregate_phylogenies_polars :
        Aggregates a list of separate phylogeny dataframes.
    """
    if isinstance(partitioned_by, (str, pl.Expr)):
        partitioned_by = [partitioned_by]
    else:
        partitioned_by = list(partitioned_by)
    if not partitioned_by:
        raise ValueError("partitioned_by must specify at least one column")

    is_eager = isinstance(phylogeny_df, pl.DataFrame)
    lf = phylogeny_df.lazy()
    schema = lf.collect_schema()
    schema_names = schema.names()

    if "ancestor_list" in schema_names:
        raise NotImplementedError(
            "ancestor_list column not supported in polars implementation",
        )
    if "ancestor_id" not in schema_names:
        raise NotImplementedError(
            "ancestor_id column required in polars implementation",
        )

    key_names = [f"{_key_prefix}{i}" for i in range(len(partitioned_by))]
    key_cols = [pl.col(name) for name in key_names]

    # number partitions by order of first appearance
    lf = (
        lf.with_columns(
            [
                (pl.col(key) if isinstance(key, str) else key).alias(name)
                for key, name in zip(partitioned_by, key_names)
            ],
        )
        .with_row_index(_row_name)
        .with_columns(
            pl.struct(key_cols).is_first_distinct().alias(_first_name),
        )
        .with_columns(
            pl.when(pl.col(_first_name))
            .then(pl.col(_first_name).cum_sum() - 1)
            .alias(_part_name),
        )
        .with_columns(pl.col(_part_name).max().over(key_cols))
        # stable w.r.t. original row order, gathers partitions together
        .sort(_part_name, _row_name)
        # shift each partition's ids past the ids of all prior partitions
        .with_columns(
            pl.when(pl.col(_first_name))
            .then(pl.col("id").max().over(_part_name) + 1)
            .otherwise(0)
            .alias(_span_name),
        )
        .with_columns(
            (pl.col(_span_name).cum_sum() - pl.col(_span_name)).alias(
                _offset_name,
            ),
        )
        .with_columns(pl.col(_offset_name).first().over(_part_name))
        .with_columns(
            (pl.col("id") + pl.col(_offset_name)).cast(schema["id"]),
            (pl.col("ancestor_id") + pl.col(_offset_name)).cast(
                schema["ancestor_id"],
            ),
        )
        .select(schema_names)
    )

    return lf.collect() if is_eager else lf
