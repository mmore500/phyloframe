import logging
import typing

import polars as pl

from ._alifestd_coarsen_taxa_asexual import (
    _alifestd_coarsen_taxa_asexual_fast_path,
)
from ._alifestd_has_contiguous_ids_polars import (
    alifestd_has_contiguous_ids_polars,
)
from ._alifestd_is_topologically_sorted_polars import (
    alifestd_is_topologically_sorted_polars,
)
from ._alifestd_topological_sensitivity_warned_polars import (
    alifestd_topological_sensitivity_warned_polars,
)
from ._alifestd_try_add_ancestor_id_col_polars import (
    alifestd_try_add_ancestor_id_col_polars,
)
from ._alifestd_try_add_ancestor_list_col_polars import (
    alifestd_try_add_ancestor_list_col_polars,
)


def alifestd_coarsen_taxa_polars_make_agg(
    phylogeny_df: pl.DataFrame,
    default_agg: typing.Callable[[pl.Expr], pl.Expr] = pl.Expr.first,
) -> typing.Dict[str, pl.Expr]:
    """Build per-column aggregation expressions for polars taxa coarsening.

    Parameters
    ----------
    phylogeny_df : polars.DataFrame or polars.LazyFrame
        Input phylogeny table.
    default_agg : Callable[[polars.Expr], polars.Expr], default Expr.first
        Aggregation to apply to any column not in the hard-coded overrides,
        called with that column's expression (e.g., `pl.Expr.max`, or
        `lambda expr: expr.drop_nulls().first()`).

    Returns
    -------
    Dict[str, polars.Expr]
        Mapping of column name to aggregation expression. Three columns are
        overridden as follows:

        - "destruction_time": `last`
        - "is_root": `first`
        - "origin_time": `first`

        Columns named

        - "ancestor_id"
        - "ancestor_list"
        - "branch_length"
        - "edge_length"
        - "id"
        - "is_leaf"

        will be excluded from the result. All other (non-excluded) columns use
        `default_agg`.

    See Also
    --------
    alifestd_coarsen_taxa_asexual_make_agg :
        Pandas-based implementation.
    """
    overrides = {
        "destruction_time": pl.Expr.last,
        "is_root": pl.Expr.first,
        "origin_time": pl.Expr.first,
    }
    return {
        col: overrides.get(col, default_agg)(pl.col(col))
        for col in phylogeny_df.lazy().collect_schema().names()
        if col
        not in (
            "ancestor_id",
            "ancestor_list",
            "branch_length",
            "edge_length",
            "id",
            "is_leaf",
        )
    }


def _differs_from_parent(
    col: str,
    dtype: pl.DataType,
    *,
    nulls_equal: typing.Optional[bool],
    nans_equal: typing.Optional[bool],
) -> pl.Expr:
    """Implementation detail for `alifestd_coarsen_taxa_polars`.

    Does node's `col` value differ from its parent's?
    """
    value = pl.col(col)
    differs = value.ne_missing(value.gather(pl.col("ancestor_id")))
    if not nulls_equal:  # None only if no nulls present
        differs |= value.is_null()
    if dtype.is_float() and not nans_equal:  # None only if no NaN present
        differs |= value.is_nan().fill_null(False)
    return differs


@alifestd_topological_sensitivity_warned_polars(
    insert=False,
    delete=True,
    update=True,
)
def alifestd_coarsen_taxa_polars(
    phylogeny_df: pl.DataFrame,
    *,
    agg: typing.Optional[typing.Mapping[str, pl.Expr]] = None,
    by: typing.Union[str, typing.Sequence[str]],
    nans_equal: typing.Optional[bool] = None,
    nulls_equal: typing.Optional[bool] = None,
) -> pl.DataFrame:
    """Condense consecutive phylogeny nodes sharing identical trait values,
    according to values in `by` column(s).

    Each condensed taxon is represented by its founder (i.e., its topmost
    node), whose id it takes.

    Parameters
    ----------
    phylogeny_df : polars.DataFrame or polars.LazyFrame
        The phylogeny as a dataframe in alife standard format.

        Must represent an asexual phylogeny with contiguous ids and
        topologically sorted rows.
    agg : Mapping[str, polars.Expr], optional
        Aggregation expression for each output column, keyed by output column
        name. Columns not included are dropped. Members of each condensed
        taxon are aggregated in topological order, so `first` gives the
        founder's value. Note that, unlike in pandas, `first` and `last` do
        not skip nulls; use, e.g., `pl.col(col).drop_nulls().first()`.

        If None, defaults to `alifestd_coarsen_taxa_polars_make_agg`.
    by : str or Sequence[str]
        Column(s) whose values must be shared for consecutive nodes to be
        condensed.
    nans_equal : bool, optional
        Whether NaN values in `by` columns compare equal to one another. If
        False, taxa with NaN `by` values are never condensed, matching the
        pandas implementation. If None (default), NaN values in `by` columns
        raise ValueError.
    nulls_equal : bool, optional
        Whether null values in `by` columns compare equal to one another. If
        False, taxa with null `by` values are never condensed. If None
        (default), null values in `by` columns raise ValueError.

    Returns
    -------
    polars.DataFrame or polars.LazyFrame
        The coarsened phylogeny, with an "is_root" column. Ids are retained
        from taxon founders, so may not be contiguous.

    Raises
    ------
    NotImplementedError
        If `phylogeny_df` is sexual, has non-contiguous ids, or is not
        topologically sorted.
    ValueError
        If `agg` would overwrite "id", "ancestor_id", or "ancestor_list", or
        if `by` columns contain NaN or null values and `nans_equal` or
        `nulls_equal`, respectively, is None.

    See Also
    --------
    alifestd_coarsen_taxa_asexual :
        Pandas-based implementation.
    alifestd_coarsen_taxa_polars_make_agg :
        Helper function to generate default `agg` dict, which may be customized
        before being passed to `alifestd_coarsen_taxa_polars`.
    """
    logging.info(
        "- alifestd_coarsen_taxa_polars: adding ancestor_id col...",
    )
    phylogeny_df = alifestd_try_add_ancestor_id_col_polars(phylogeny_df)

    schema = phylogeny_df.lazy().collect_schema()
    if "ancestor_id" not in schema.names():
        raise NotImplementedError("sexual phylogenies not supported")

    if "is_root" not in schema.names():
        phylogeny_df = phylogeny_df.with_columns(
            is_root=pl.col("id") == pl.col("ancestor_id"),
        )
        schema = phylogeny_df.lazy().collect_schema()

    if agg is None:
        agg = alifestd_coarsen_taxa_polars_make_agg(phylogeny_df)

    for col in "id", "ancestor_id", "ancestor_list":
        if col in agg:
            raise ValueError(f"agg for `{col}` column may not be overwritten")

    if isinstance(by, str):
        by = (by,)

    if nulls_equal is None and (
        phylogeny_df.lazy()
        .select(pl.any_horizontal([pl.col(col).is_null().any() for col in by]))
        .collect()
        .item()
    ):
        raise ValueError(
            "null values in `by` columns; "
            "set `nulls_equal` to specify how they compare",
        )

    float_by = [col for col in by if schema[col].is_float()]
    if (
        nans_equal is None
        and float_by
        and (
            phylogeny_df.lazy()
            .select(
                pl.any_horizontal(
                    [pl.col(col).is_nan().any() for col in float_by]
                ),
            )
            .collect()
            .item()
        )
    ):
        raise ValueError(
            "NaN values in `by` columns; "
            "set `nans_equal` to specify how they compare",
        )

    if not phylogeny_df.lazy().limit(1).collect().is_empty():
        if not alifestd_has_contiguous_ids_polars(phylogeny_df):
            raise NotImplementedError("non-contiguous ids not supported")

        if not alifestd_is_topologically_sorted_polars(phylogeny_df):
            raise NotImplementedError(
                "non-topologically-sorted data not supported",
            )

    logging.info(
        "- alifestd_coarsen_taxa_polars: marking taxon founders...",
    )
    founder_df = (
        phylogeny_df.lazy()
        .select(
            "ancestor_id",
            "is_root",
            is_taxon_founder=pl.any_horizontal(
                [
                    _differs_from_parent(
                        col,
                        schema[col],
                        nans_equal=nans_equal,
                        nulls_equal=nulls_equal,
                    )
                    for col in by
                ],
            )
            | pl.col("is_root"),
        )
        .collect()
    )

    logging.info(
        "- alifestd_coarsen_taxa_polars: finding taxon founders...",
    )
    ancestor_ids, founder_ids = _alifestd_coarsen_taxa_asexual_fast_path(
        founder_df["ancestor_id"].to_numpy().copy(),  # remove read-only flag
        founder_df["is_taxon_founder"].to_numpy(),
        founder_df["is_root"].to_numpy(),
    )

    logging.info("- alifestd_coarsen_taxa_polars: condensing taxa...")
    phylogeny_df = (
        phylogeny_df.with_columns(
            pl.Series("ancestor_id", ancestor_ids).cast(schema["ancestor_id"]),
            pl.Series("_alifestd_coarsen_taxa_polars_founder_id", founder_ids),
        )
        .group_by(
            "_alifestd_coarsen_taxa_polars_founder_id", maintain_order=True
        )
        .agg(
            pl.col("id").first(),
            pl.col("ancestor_id").first(),
            **agg,
        )
        .drop("_alifestd_coarsen_taxa_polars_founder_id")
    )

    if "ancestor_list" in schema.names():
        phylogeny_df = alifestd_try_add_ancestor_list_col_polars(phylogeny_df)

    return phylogeny_df
