import argparse
import functools
import logging
import os
import typing

import joinem
from joinem._dataframe_cli import _add_parser_base, _run_dataframe_cli
import numpy as np
import polars as pl

from .._auxlib._begin_prod_logging import begin_prod_logging
from .._auxlib._format_cli_description import format_cli_description
from .._auxlib._get_phyloframe_version import get_phyloframe_version
from .._auxlib._log_context_duration import log_context_duration
from ._alifestd_has_contiguous_ids_polars import (
    alifestd_has_contiguous_ids_polars,
)
from ._alifestd_is_topologically_sorted_polars import (
    alifestd_is_topologically_sorted_polars,
)
from ._alifestd_mask_persistence_asexual import (
    _alifestd_mask_persistence_asexual_fast_path,
)
from ._alifestd_try_add_ancestor_id_col_polars import (
    alifestd_try_add_ancestor_id_col_polars,
)


def alifestd_mask_persistence_polars(
    phylogeny_df: pl.DataFrame,
    *,
    candidate_mask: typing.Union[str, pl.Expr],
    criterion_cherry: typing.Optional[
        typing.Union[str, pl.Expr]
    ] = "origin_time",
    criterion_trunk: typing.Optional[
        typing.Union[str, pl.Expr]
    ] = "origin_time",
    extant_mask: typing.Union[str, pl.Expr],
    mark_as: str = "alifestd_mask_persistence_polars",
) -> pl.DataFrame:
    """Mark candidate taxa that are the closest relative, among candidates,
    of at least one extant taxon.

    Useful to identify which earlier-sampled taxa (candidates) have
    lineages that persist among later-sampled taxa (extant), where
    candidates may not be direct ancestors of extant taxa (e.g., both are
    leaves).

    Distance between an extant taxon and a candidate is measured via their
    most recent common ancestor (MRCA), as the trunk distance along the
    extant taxon's lineage back to the MRCA, plus the cherry distance from
    the MRCA out to the candidate:

        distance = (criterion_trunk[extant] - criterion_trunk[mrca])
                 + (criterion_cherry[candidate] - criterion_cherry[mrca])

    If several candidates tie as an extant taxon's closest, all are marked.
    Extant taxa that share no common ancestor with any candidate do not
    mark any candidate. A taxon that is both extant and a candidate is its
    own closest relative.

    Only supports asexual phylogenies.

    Parameters
    ----------
    phylogeny_df : polars.DataFrame or polars.LazyFrame
        The phylogeny as a dataframe in alife standard format.

        Must represent an asexual phylogeny with contiguous,
        topologically sorted ids.
    candidate_mask : str or polars.Expr
        Column name or boolean expression marking taxa eligible to be
        marked as closest relatives, e.g., leaves within an earlier time
        window. May include inner nodes. Null values are not supported.
    criterion_cherry : str or polars.Expr, optional, default "origin_time"
        Column name or expression used to measure cherry distance, from
        MRCA to candidate, e.g., "origin_time" for elapsed time or
        "node_depth" for number of edges (see
        `alifestd_mark_node_depth_polars`). Values must be finite, and must
        not decrease from ancestor to descendant.

        If None, cherry distance is not counted; see Notes.
    criterion_trunk : str or polars.Expr, optional, default "origin_time"
        Column name or expression used to measure trunk distance, from MRCA
        to extant taxon, with requirements as for `criterion_cherry`.

        If None, trunk distance is not counted; see Notes.
    extant_mask : str or polars.Expr
        Column name or boolean expression marking taxa whose closest
        relatives are sought, e.g., leaves after a time threshold. May
        include inner nodes. Null values are not supported.
    mark_as : str, default "alifestd_mask_persistence_polars"
        Column name for the boolean mask.

    Raises
    ------
    NotImplementedError
        If `phylogeny_df` has no "ancestor_id" column, if ids are
        non-contiguous or not topologically sorted, if `extant_mask` or
        `candidate_mask` has null values, or if `criterion_cherry` or
        `criterion_trunk` values are null, non-finite, or decrease from
        ancestor to descendant (see
        `alifestd_coerce_chronological_consistency_polars`).
    ValueError
        If `criterion_cherry` or `criterion_trunk` is not a column in
        `phylogeny_df`, or if both are None.

    Returns
    -------
    polars.DataFrame
        The phylogeny with an added boolean mask column, a subset of
        `candidate_mask`.

    See Also
    --------
    alifestd_find_pair_distance_polars :
        Calculates distance between a single pair of taxa.
    alifestd_mark_sample_tips_lineage_polars :
        Marks leaves closest to the lineage of a single target leaf.
    alifestd_mask_persistence_asexual :
        Pandas-based implementation.
    alifestd_prune_extinct_lineages_polars :
        Drops taxa without marked descendants, e.g., pass `mark_as` column
        as `criterion` to keep only lineages of marked candidates.

    Notes
    -----
    Which candidates are closest depends on how trunk and cherry distances
    are counted:

    - with both (default), candidates are closest by total distance, as in
      `alifestd_find_pair_distance_polars`;
    - with `criterion_cherry` None, candidates are closest by most recent
      common ancestor, whatever their own branch lengths, so all candidates
      sharing an extant taxon's most recent common ancestor with any
      candidate are marked; and
    - with `criterion_trunk` None, candidates are closest by how far they
      sit from an extant taxon's lineage, however long ago they diverged
      from it.

    Examples
    --------
    Take phylogeny `df` in working format. To mark leaves from times 40 to
    60 with lineages that persist among leaves after time 100:

    >>> is_leaf = pl.col("is_leaf")
    >>> origin_time = pl.col("origin_time")
    >>> candidate_mask = is_leaf & origin_time.is_between(40, 60)
    >>> extant_mask = is_leaf & (origin_time > 100)
    >>> persistence_df = alifestd_mark_leaves_polars(df).pipe(
    ...     alifestd_mask_persistence_polars,
    ...     candidate_mask=candidate_mask,
    ...     extant_mask=extant_mask,
    ...     mark_as="is_persistent",
    ... )

    To then drop all but these leaves' lineages:

    >>> persistence_df = alifestd_prune_extinct_lineages_polars(
    ...     persistence_df, criterion="is_persistent"
    ... )

    To instead mark by most recent common ancestor, whatever leaves' own
    branch lengths:

    >>> persistence_df = alifestd_mark_leaves_polars(df).pipe(
    ...     alifestd_mask_persistence_polars,
    ...     candidate_mask=candidate_mask,
    ...     criterion_cherry=None,
    ...     extant_mask=extant_mask,
    ...     mark_as="is_persistent",
    ... )
    """
    if isinstance(extant_mask, str):
        extant_mask = pl.col(extant_mask)
    if isinstance(candidate_mask, str):
        candidate_mask = pl.col(candidate_mask)

    if criterion_cherry is None and criterion_trunk is None:
        raise ValueError(
            "at least one of criterion_cherry and criterion_trunk required",
        )
    phylogeny_df = alifestd_try_add_ancestor_id_col_polars(phylogeny_df)
    schema_names = phylogeny_df.lazy().collect_schema().names()
    for criterion in (criterion_cherry, criterion_trunk):
        if isinstance(criterion, str) and criterion not in schema_names:
            raise ValueError(
                f"criterion column {criterion!r} not found in phylogeny_df",
            )
    if "ancestor_id" not in schema_names:
        raise NotImplementedError(
            "alifestd_mask_persistence_polars only supports asexual phylogenies.",
        )
    if phylogeny_df.lazy().limit(1).collect().is_empty():
        return phylogeny_df.with_columns(
            pl.lit(False, dtype=pl.Boolean).alias(mark_as),
        )
    if not alifestd_has_contiguous_ids_polars(phylogeny_df):
        raise NotImplementedError("non-contiguous ids not supported")
    if not alifestd_is_topologically_sorted_polars(phylogeny_df):
        raise NotImplementedError(
            "non-topologically-sorted data not supported",
        )

    criteria = {
        name: pl.lit(0.0)
        if criterion is None
        else pl.col(criterion)
        if isinstance(criterion, str)
        else criterion
        for name, criterion in (
            ("criterion_cherry", criterion_cherry),
            ("criterion_trunk", criterion_trunk),
        )
    }
    selected = (
        phylogeny_df.lazy()
        .select(
            pl.col("ancestor_id"),
            extant_mask=extant_mask.cast(pl.Boolean),
            candidate_mask=candidate_mask.cast(pl.Boolean),
            **{k: v.cast(pl.Float64) for k, v in criteria.items()},
        )
        .collect()
    )
    for name in selected.columns[1:]:
        if selected[name].has_nulls():
            raise NotImplementedError(f"null {name} values not supported")
    ancestor_ids = selected["ancestor_id"].to_numpy()
    for name in "criterion_cherry", "criterion_trunk":
        values = selected[name].to_numpy()
        if not np.isfinite(values).all():
            raise NotImplementedError(
                f"non-finite {name} values not supported",
            )
        if (values < values[ancestor_ids]).any():
            raise NotImplementedError(
                f"{name} values decreasing from ancestor to descendant "
                "not supported",
            )

    result = _alifestd_mask_persistence_asexual_fast_path(
        ancestor_ids,
        selected["extant_mask"].to_numpy(),
        selected["candidate_mask"].to_numpy(),
        selected["criterion_cherry"].to_numpy(),
        selected["criterion_trunk"].to_numpy(),
    )
    return phylogeny_df.with_columns(pl.Series(name=mark_as, values=result))


_raw_description = f"""\
{os.path.basename(__file__)} | \
(phyloframe v{get_phyloframe_version()}/joinem v{joinem.__version__})

Mark candidate taxa that are the closest relative, among candidates, of at
least one extant taxon.

Useful to identify which earlier-sampled taxa (candidates) have lineages
that persist among later-sampled taxa (extant).

Data is assumed to be in alife standard format.
Only supports asexual phylogenies.

Additional Notes
================
- Requires contiguous, topologically sorted ids.

- Mask and criterion arguments are evaluated as Python expressions, with
polars available as `pl`, e.g.,
`--extant-mask 'pl.col("is_leaf") & (pl.col("origin_time") > 100)'`.
Column names must be quoted, e.g., `--criterion-trunk '"node_depth"'`.

- Uses `--criterion-cherry` and `--criterion-trunk` (default:
'origin_time') to measure distance from the most recent common ancestor of
an extant taxon and a candidate out to the candidate (cherry) and to the
extant taxon (trunk).

- Pass `--criterion-cherry None` to mark candidates by most recent common
ancestor, whatever their own branch lengths, or `--criterion-trunk None` to
mark candidates by how far they sit from an extant taxon's lineage.

- Use `--eager-read` if modifying data file inplace.

- This CLI entrypoint is experimental and may be subject to change.

See Also
========
phyloframe.legacy._alifestd_mark_leaves_polars :
    Adds `is_leaf` column, e.g., to define mask columns.

phyloframe.legacy._alifestd_prune_extinct_lineages_polars :
    Drops taxa without marked descendants, e.g., pass `--mark-as` column as
    `--criterion` to keep only lineages of marked candidates.
"""


def _create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        add_help=False,
        allow_abbrev=False,
        description=format_cli_description(_raw_description),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser = _add_parser_base(
        parser=parser,
        dfcli_module=("phyloframe.legacy._alifestd_mask_persistence_polars"),
        dfcli_version=get_phyloframe_version(),
    )
    parser.add_argument(
        "--candidate-mask",
        required=True,
        type=str,
        help=(
            "Python expression evaluating to a column name or boolean polars "
            "expression marking taxa eligible to be marked as closest "
            "relatives, e.g., leaves within an earlier time window, with "
            "polars available as `pl`."
        ),
    )
    parser.add_argument(
        "--criterion-cherry",
        default='"origin_time"',
        type=str,
        help=(
            "Python expression evaluating to a column name or polars "
            "expression used to measure distance from MRCA to candidate, or "
            "None to not count, with polars available as `pl` "
            '(default: "origin_time").'
        ),
    )
    parser.add_argument(
        "--criterion-trunk",
        default='"origin_time"',
        type=str,
        help=(
            "Python expression evaluating to a column name or polars "
            "expression used to measure distance from MRCA to extant taxon, "
            "or None to not count, with polars available as `pl` "
            '(default: "origin_time").'
        ),
    )
    parser.add_argument(
        "--extant-mask",
        required=True,
        type=str,
        help=(
            "Python expression evaluating to a column name or boolean polars "
            "expression marking taxa whose closest relatives are sought, "
            "e.g., leaves after a time threshold, with polars available as "
            "`pl`."
        ),
    )
    parser.add_argument(
        "--mark-as",
        default="alifestd_mask_persistence_polars",
        type=str,
        help=(
            "Column name for the boolean mask "
            "(default: alifestd_mask_persistence_polars)."
        ),
    )
    return parser


if __name__ == "__main__":
    begin_prod_logging()

    parser = _create_parser()
    args, __ = parser.parse_known_args()

    candidate_mask, criterion_cherry, criterion_trunk, extant_mask = (
        eval(arg, {"pl": pl})  # nosec B307
        for arg in (
            args.candidate_mask,
            args.criterion_cherry,
            args.criterion_trunk,
            args.extant_mask,
        )
    )

    try:
        with log_context_duration(
            "phyloframe.legacy._alifestd_mask_persistence_polars",
            logging.info,
        ):
            _run_dataframe_cli(
                base_parser=parser,
                output_dataframe_op=functools.partial(
                    alifestd_mask_persistence_polars,
                    candidate_mask=candidate_mask,
                    criterion_cherry=criterion_cherry,
                    criterion_trunk=criterion_trunk,
                    extant_mask=extant_mask,
                    mark_as=args.mark_as,
                ),
            )
    except NotImplementedError as e:
        logging.error(
            "- polars op not yet implemented, use pandas op instead",
        )
        raise e
