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
from ._alifestd_mark_lineage_cummax_asexual import (
    _alifestd_mark_lineage_cummax_asexual_fast_path,
)
from ._alifestd_mark_lineage_cummin_asexual import (
    _alifestd_mark_lineage_cummin_asexual_fast_path,
)
from ._alifestd_try_add_ancestor_id_col_polars import (
    alifestd_try_add_ancestor_id_col_polars,
)


def _rank_pairs(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Implementation detail for `alifestd_mask_persistence_polars`.

    Returns float ranks of pairs ordered by `first` value, then by `second`
    value, from 1 for the lowest, with equal pairs sharing a rank.
    """
    return (
        pl.DataFrame({"first": first, "second": second})
        .select(pl.struct("first", "second").rank("dense"))
        .to_series()
        .to_numpy()
        .astype(np.float64)
    )


def _alifestd_mask_persistence_polars_fast_path(
    ancestor_ids: np.ndarray,
    extant_mask: np.ndarray,
    candidate_mask: np.ndarray,
    cherry_values: np.ndarray,
    trunk_values: np.ndarray,
    tiebreak_values: np.ndarray,
) -> np.ndarray:
    """Implementation detail for `alifestd_mask_persistence_polars`.

    Assumes contiguous, topologically sorted ids, with finite float values,
    of which `cherry_values` and `trunk_values` do not decrease from
    ancestor to descendant.
    """
    # rank by cherry value, then by tiebreak value; the lowest-ranked
    # candidates in each node's clade are the clade's candidates closest to
    # any taxon outside the clade
    candidate_ranks = _rank_pairs(cherry_values, tiebreak_values)
    clade_candidate_ranks = _alifestd_mark_lineage_cummin_asexual_fast_path(
        ancestor_ids,
        np.where(candidate_mask, candidate_ranks, np.nan),
        True,  # reverse
        True,  # skipna
    )
    has_candidate = ~np.isnan(clade_candidate_ranks)
    # id of one of each clade's lowest-ranked candidates, to look up their
    # shared cherry and tiebreak values
    rank_ids = np.empty(len(ancestor_ids) + 1, dtype=np.int64)
    rank_ids[candidate_ranks.astype(np.int64)] = np.arange(len(ancestor_ids))
    clade_candidate_ids = rank_ids[
        np.where(has_candidate, clade_candidate_ranks, 1).astype(np.int64)
    ]
    # distance via an mrca is trunk[extant] - trunk[mrca] + cherry[candidate]
    # - cherry[mrca]; ranking each node as an mrca, by the terms that do not
    # depend on the extant taxon and then by tiebreak value, the lowest rank
    # along a taxon's lineage is at its mrca with its closest candidates
    # (where a clade's lowest-ranked candidate shares a later mrca with the
    # taxon, this only overestimates distance, as values do not decrease)
    mrca_ranks = np.where(
        has_candidate,
        _rank_pairs(
            cherry_values[clade_candidate_ids]
            - (trunk_values + cherry_values),
            tiebreak_values[clade_candidate_ids],
        ),
        np.nan,
    )
    best_ranks = _alifestd_mark_lineage_cummin_asexual_fast_path(
        ancestor_ids, mrca_ranks, False, True
    )
    # a node is mrca of an extant taxon and its closest candidates if its
    # rank is the best along that taxon's lineage; no taxon in a node's
    # clade can have a best rank exceeding the node's own rank
    is_closest_mrca = mrca_ranks == (
        _alifestd_mark_lineage_cummax_asexual_fast_path(
            ancestor_ids,
            np.where(extant_mask, best_ranks, np.nan),
            True,  # reverse
            True,  # skipna
        )
    )
    # a candidate is a closest relative if it is lowest-ranked among
    # candidates in the clade of such an mrca; no ancestor's clade can have
    # a lowest rank exceeding the candidate's own
    return candidate_mask & (
        candidate_ranks
        == _alifestd_mark_lineage_cummax_asexual_fast_path(
            ancestor_ids,
            np.where(is_closest_mrca, clade_candidate_ranks, np.nan),
            False,  # reverse
            True,  # skipna
        )
    )


def alifestd_mask_persistence_polars(
    phylogeny_df: pl.DataFrame,
    *,
    candidate_mask: typing.Union[str, pl.Expr],
    extant_mask: typing.Union[str, pl.Expr],
    criterion_cherry: typing.Union[str, pl.Expr] = "origin_time",
    criterion_tiebreak: typing.Union[str, pl.Expr] = pl.col(
        "origin_time"
    ).rank("ordinal"),
    criterion_trunk: typing.Union[str, pl.Expr] = "origin_time",
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

    If several candidates tie as an extant taxon's closest, the one with
    the lowest `criterion_tiebreak` value is marked. Extant taxa that share
    no common ancestor with any candidate do not mark any candidate. A
    taxon that is both extant and a candidate is its own closest relative.

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
    extant_mask : str or polars.Expr
        Column name or boolean expression marking taxa whose closest
        relatives are sought, e.g., leaves after a time threshold. May
        include inner nodes. Null values are not supported.
    criterion_cherry : str or polars.Expr, default "origin_time"
        Column name or expression used to measure cherry distance, from
        MRCA to candidate, e.g., "origin_time" for elapsed time or
        "node_depth" for number of edges (see
        `alifestd_mark_node_depth_polars`). Values must be finite, and must
        not decrease from ancestor to descendant.

        Pass a constant, e.g., `pl.lit(0)`, to not count cherry distance;
        see Notes.
    criterion_tiebreak : str or polars.Expr, optional
        Column name or expression used to choose among candidates tied as
        an extant taxon's closest, taking the candidate with the lowest
        value. Values must be finite. Candidates also tied by
        `criterion_tiebreak` value are all marked.

        Defaults to ordinal rank of "origin_time", taking the earliest tied
        candidate, or the first in row order among equally early candidates,
        so that each extant taxon marks at most one candidate.

        Pass a constant, e.g., `pl.lit(0)`, to mark all tied candidates.
    criterion_trunk : str or polars.Expr, default "origin_time"
        Column name or expression used to measure trunk distance, from MRCA
        to extant taxon, with requirements as for `criterion_cherry`.

        Pass a constant, e.g., `pl.lit(0)`, to not count trunk distance;
        see Notes.
    mark_as : str, default "alifestd_mask_persistence_polars"
        Column name for the boolean mask.

    Raises
    ------
    NotImplementedError
        If `phylogeny_df` has no "ancestor_id" column, if ids are
        non-contiguous or not topologically sorted, if `extant_mask` or
        `candidate_mask` has null values, if criterion values are null or
        non-finite, or if `criterion_cherry` or `criterion_trunk` values
        decrease from ancestor to descendant (see
        `alifestd_coerce_chronological_consistency_polars`).
    ValueError
        If a criterion is given as a column name not in `phylogeny_df`.

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
    alifestd_prune_extinct_lineages_polars :
        Drops taxa without marked descendants, e.g., pass `mark_as` column
        as `criterion` to keep only lineages of marked candidates.

    Notes
    -----
    Which candidates are closest depends on how trunk and cherry distances
    are counted:

    - with both (default), candidates are closest by total distance, as in
      `alifestd_find_pair_distance_polars`;
    - with constant `criterion_cherry`, candidates are closest by most
      recent common ancestor, whatever their own branch lengths, so all
      candidates sharing an extant taxon's most recent common ancestor with
      any candidate tie, and are chosen among by `criterion_tiebreak`; and
    - with constant `criterion_trunk`, candidates are closest by how far
      they sit from an extant taxon's lineage, however long ago they
      diverged from it.

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
    ...     criterion_cherry=pl.lit(0),
    ...     extant_mask=extant_mask,
    ...     mark_as="is_persistent",
    ... )
    """
    if isinstance(extant_mask, str):
        extant_mask = pl.col(extant_mask)
    if isinstance(candidate_mask, str):
        candidate_mask = pl.col(candidate_mask)

    phylogeny_df = alifestd_try_add_ancestor_id_col_polars(phylogeny_df)
    schema_names = phylogeny_df.lazy().collect_schema().names()
    criteria = {
        "criterion_cherry": criterion_cherry,
        "criterion_tiebreak": criterion_tiebreak,
        "criterion_trunk": criterion_trunk,
    }
    for name, criterion in criteria.items():
        if isinstance(criterion, str) and criterion not in schema_names:
            raise ValueError(
                f"{name} column {criterion!r} not found in phylogeny_df",
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

    selected = (
        phylogeny_df.lazy()
        .select(
            pl.col("ancestor_id"),
            extant_mask=extant_mask.cast(pl.Boolean),
            candidate_mask=candidate_mask.cast(pl.Boolean),
            **{
                name: (
                    pl.col(criterion)
                    if isinstance(criterion, str)
                    else criterion
                ).cast(pl.Float64)
                for name, criterion in criteria.items()
            },
        )
        .collect()
    )
    for name in selected.columns[1:]:
        if selected[name].has_nulls():
            raise NotImplementedError(f"null {name} values not supported")
    ancestor_ids = selected["ancestor_id"].to_numpy()
    for name in criteria:
        values = selected[name].to_numpy()
        if not np.isfinite(values).all():
            raise NotImplementedError(
                f"non-finite {name} values not supported",
            )
        if (
            name != "criterion_tiebreak"
            and (values < values[ancestor_ids]).any()
        ):
            raise NotImplementedError(
                f"{name} values decreasing from ancestor to descendant "
                "not supported",
            )

    result = _alifestd_mask_persistence_polars_fast_path(
        ancestor_ids,
        selected["extant_mask"].to_numpy(),
        selected["candidate_mask"].to_numpy(),
        selected["criterion_cherry"].to_numpy(),
        selected["criterion_trunk"].to_numpy(),
        selected["criterion_tiebreak"].to_numpy(),
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

- Pass `--criterion-cherry 'pl.lit(0)'` to mark candidates by most recent
common ancestor, whatever their own branch lengths, or
`--criterion-trunk 'pl.lit(0)'` to mark candidates by how far they sit from
an extant taxon's lineage.

- Among candidates tied as an extant taxon's closest, the one with the
lowest `--criterion-tiebreak` value is marked (default: earliest
'origin_time', then first in row order). Pass
`--criterion-tiebreak 'pl.lit(0)'` to mark all tied candidates.

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
        "--criterion-cherry",
        default='"origin_time"',
        type=str,
        help=(
            "Python expression evaluating to a column name or polars "
            "expression used to measure distance from MRCA to candidate, "
            "with polars available as `pl` "
            '(default: "origin_time").'
        ),
    )
    parser.add_argument(
        "--criterion-tiebreak",
        default='pl.col("origin_time").rank("ordinal")',
        type=str,
        help=(
            "Python expression evaluating to a column name or polars "
            "expression used to choose among tied candidates, taking the "
            "lowest, with polars available as `pl` "
            '(default: pl.col("origin_time").rank("ordinal")).'
        ),
    )
    parser.add_argument(
        "--criterion-trunk",
        default='"origin_time"',
        type=str,
        help=(
            "Python expression evaluating to a column name or polars "
            "expression used to measure distance from MRCA to extant taxon, "
            "with polars available as `pl` "
            '(default: "origin_time").'
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

    kwargs = {
        name: eval(getattr(args, name), {"pl": pl})  # nosec B307
        for name in (
            "candidate_mask",
            "extant_mask",
            "criterion_cherry",
            "criterion_tiebreak",
            "criterion_trunk",
        )
    }

    with log_context_duration(
        "phyloframe.legacy._alifestd_mask_persistence_polars",
        logging.info,
    ):
        _run_dataframe_cli(
            base_parser=parser,
            output_dataframe_op=functools.partial(
                alifestd_mask_persistence_polars,
                mark_as=args.mark_as,
                **kwargs,
            ),
        )
