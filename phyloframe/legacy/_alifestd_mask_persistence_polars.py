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
from .._auxlib._cmp import cmp
from .._auxlib._format_cli_description import format_cli_description
from .._auxlib._get_phyloframe_version import get_phyloframe_version
from .._auxlib._jit import jit
from .._auxlib._log_context_duration import log_context_duration
from ._alifestd_has_contiguous_ids_polars import (
    alifestd_has_contiguous_ids_polars,
)
from ._alifestd_is_topologically_sorted_polars import (
    alifestd_is_topologically_sorted_polars,
)
from ._alifestd_mark_node_depth_polars import alifestd_mark_node_depth_polars
from ._alifestd_try_add_ancestor_id_col_polars import (
    alifestd_try_add_ancestor_id_col_polars,
)


@jit(nopython=True)
def _compare_mrcas(first: int, second: int, mrca_values: np.ndarray) -> int:
    """Implementation detail for `_alifestd_mask_persistence_fast_path`.

    Returns -1, 0, or 1 if node `first` is closer than, as close as, or
    further than node `second`, as an mrca with its clade's closest
    candidate, comparing fields of their `mrca_values` in order. Node -1 is
    closer than any other, and node n further.
    """
    if first == second:  # same node
        return 0
    elif first == -1 or second == len(mrca_values):  # sentinel: closer
        return -1
    elif second == -1 or first == len(mrca_values):  # sentinel: further
        return 1
    else:  # first field differing
        return cmp(mrca_values[first], mrca_values[second])


@jit(nopython=True)
def _alifestd_mask_persistence_fast_path(
    ancestor_ids: np.ndarray,
    candidate_mask: np.ndarray,
    extant_mask: np.ndarray,
    candidate_ranks: np.ndarray,
    cherry_values: np.ndarray,
    trunk_legs: np.ndarray,
    tiebreak_values: np.ndarray,
) -> np.ndarray:
    """Implementation detail for `alifestd_mask_persistence_polars`.

    Assumes contiguous, topologically sorted ids. Values have a row per
    taxon, with columns compared in order; `trunk_legs` are distances from
    the largest trunk values, and `candidate_ranks` rank taxa by cherry
    then tiebreak values.
    """
    n = len(ancestor_ids)
    num_fields = cherry_values.shape[1]

    # sweep up, for each node's clade: the closest candidate to taxa outside
    # it; distance from the node, as mrca, to that candidate, then its
    # tiebreak; and the mrca within it that is furthest among those closest
    # to each extant taxon within it (-1 if none, n if without candidates)
    clade_candidate_ids = np.where(candidate_mask, np.arange(n), -1)
    mrca_values = np.column_stack(
        (trunk_legs + cherry_values, tiebreak_values)
    )
    clade_mrca_ids = np.full(n, -1)
    for id_ in range(n - 1, -1, -1):
        candidate_id = clade_candidate_ids[id_]
        mrca_id = id_ if candidate_id != -1 else n
        if candidate_id != -1:
            mrca_values[id_, :num_fields] = trunk_legs[id_] + (
                cherry_values[candidate_id] - cherry_values[id_]
            )
            mrca_values[id_, num_fields:] = tiebreak_values[candidate_id]
        if extant_mask[id_] or (
            _compare_mrcas(mrca_id, clade_mrca_ids[id_], mrca_values) == -1
        ):
            clade_mrca_ids[id_] = mrca_id

        ancestor_id = ancestor_ids[id_]
        if ancestor_id == id_:
            continue
        ancestor_mrca_id = clade_mrca_ids[ancestor_id]
        if (
            _compare_mrcas(clade_mrca_ids[id_], ancestor_mrca_id, mrca_values)
            == 1
        ):
            clade_mrca_ids[ancestor_id] = clade_mrca_ids[id_]
        ancestor_candidate_id = clade_candidate_ids[ancestor_id]
        if candidate_id != -1 and (
            ancestor_candidate_id == -1
            or candidate_ranks[candidate_id]
            < candidate_ranks[ancestor_candidate_id]
        ):
            clade_candidate_ids[ancestor_id] = candidate_id

    # sweep down, for each node's lineage: the closest mrca along it, and
    # the highest rank sought of candidates by mrcas closest to extant taxa
    lineage_mrca_ids = np.empty(n, dtype=np.int64)
    lineage_candidate_ranks = np.full(n, -1)
    for id_ in range(n):
        mrca_id = id_ if clade_candidate_ids[id_] != -1 else n
        lineage_mrca_ids[id_] = mrca_id

        ancestor_id = ancestor_ids[id_]
        if ancestor_id != id_:
            lineage_candidate_ranks[id_] = lineage_candidate_ranks[ancestor_id]
            ancestor_mrca_id = lineage_mrca_ids[ancestor_id]
            if _compare_mrcas(ancestor_mrca_id, mrca_id, mrca_values) == -1:
                lineage_mrca_ids[id_] = ancestor_mrca_id

        if (
            mrca_id != n
            and _compare_mrcas(mrca_id, lineage_mrca_ids[id_], mrca_values)
            == 0
            and _compare_mrcas(mrca_id, clade_mrca_ids[id_], mrca_values) == 0
        ):
            lineage_candidate_ranks[id_] = max(
                lineage_candidate_ranks[id_],
                candidate_ranks[clade_candidate_ids[id_]],
            )

    return candidate_mask & (candidate_ranks == lineage_candidate_ranks)


def alifestd_mask_persistence_polars(
    phylogeny_df: pl.DataFrame,
    *,
    candidate_mask: typing.Union[str, pl.Expr],
    extant_mask: typing.Union[str, pl.Expr],
    criterion_cherry: typing.Union[str, pl.Expr] = pl.struct(
        "origin_time", "node_depth"
    ),
    criterion_trunk: typing.Union[str, pl.Expr] = pl.struct(
        "origin_time", "node_depth"
    ),
    mark_as: str = "alifestd_mask_persistence_polars",
    criterion_tiebreak: typing.Union[str, pl.Expr] = pl.struct(
        pl.col("origin_time").rank("dense", descending=True), "id"
    ).rank("ordinal"),
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

    By default, distance is elapsed time, then number of edges among
    candidates at the same elapsed time.

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
    criterion_cherry : str or polars.Expr, default pl.struct("origin_time", "node_depth")
        Column name or expression used to measure cherry distance, from
        MRCA to candidate, e.g., "origin_time" for elapsed time or
        "node_depth" for number of edges. Values must be numeric and
        finite, and must not decrease from ancestor to descendant.

        If a struct, distances are compared by its first field, then among
        equal distances by its next field, and so on, with requirements
        applying to each field. By default, compares elapsed time, then
        number of edges.

        Column "node_depth" is calculated if not present; see
        `alifestd_mark_node_depth_polars`.

        Pass a constant, e.g., `pl.lit(0)`, to not count cherry distance;
        see Notes.
    criterion_trunk : str or polars.Expr, default pl.struct("origin_time", "node_depth")
        Column name or expression used to measure trunk distance, from MRCA
        to extant taxon, as for `criterion_cherry`. If structs, fields of
        `criterion_trunk` and `criterion_cherry` correspond in order.

        Pass a constant, e.g., `pl.lit(0)`, to not count trunk distance;
        see Notes.
    mark_as : str, default "alifestd_mask_persistence_polars"
        Column name for the boolean mask.
    criterion_tiebreak : str or polars.Expr, optional
        Column name or expression used to choose among candidates tied as
        an extant taxon's closest, taking the candidate with the lowest
        value. Values must be numeric and finite. If a struct, fields are
        compared in order. Candidates also tied by `criterion_tiebreak`
        value are all marked.

        By default, takes the latest tied candidate by "origin_time", or
        the one with the lowest id among equally late candidates, so that
        each extant taxon marks at most one candidate.

        Pass a constant, e.g., `pl.lit(0)`, to mark all tied candidates.

    Raises
    ------
    NotImplementedError
        If `phylogeny_df` has no "ancestor_id" column, if ids are
        non-contiguous or not topologically sorted, if `extant_mask` or
        `candidate_mask` has null values, if criterion values are null,
        non-numeric, or non-finite, or if `criterion_cherry` or
        `criterion_trunk` values decrease from ancestor to descendant (see
        `alifestd_coerce_chronological_consistency_polars`).
    ValueError
        If a criterion uses a column not in `phylogeny_df`.

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

    - with both (default), candidates are closest by total distance, in
      elapsed time as in `alifestd_find_pair_distance_polars`;
    - with constant `criterion_cherry`, candidates are closest by most
      recent common ancestor, whatever their own branch lengths, so all
      candidates sharing an extant taxon's most recent common ancestor with
      any candidate tie, and are chosen among by `criterion_tiebreak`; and
    - with constant `criterion_trunk`, candidates are closest by how far
      they sit from an extant taxon's lineage, however long ago they
      diverged from it.

    Marking generalizes the persistence filter of the MODES toolbox (Dolson
    et al., 2019), which keeps taxa at one time point with descendants a
    filter length later, to screen out transient taxa without the neutral
    shadow runs of evolutionary activity statistics (Bedau et al., 1998).
    With taxa before a time point as candidates, extant taxa mark their
    ancestors, as for that filter; see Examples. Otherwise, e.g., with
    only leaves as candidates, their closest relatives stand in.

    References
    ----------
    Dolson EL, Vostinar AE, Wiser MJ, and Ofria C. The MODES toolbox:
        measurements of open-ended dynamics in evolving systems. Artificial
        Life 2019, 25(1):50-73 (https://doi.org/10.1162/artl_a_00280)

    Bedau MA, Snyder E, and Packard NH. A classification of long-term
        evolutionary dynamics. In: Artificial Life VI. MIT Press, 1998,
        pp. 228-237.

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
    ...     extant_mask=extant_mask,
    ...     criterion_cherry=pl.lit(0),
    ...     mark_as="is_persistent",
    ... )

    Candidates may include inner nodes. To mark the latest ancestor before
    time 50 of each leaf after time 100, take all taxa before time 50 as
    candidates:

    >>> persistence_df = alifestd_mark_leaves_polars(df).pipe(
    ...     alifestd_mask_persistence_polars,
    ...     candidate_mask=origin_time < 50,
    ...     extant_mask=extant_mask,
    ...     mark_as="is_persistent",
    ... )

    Each such ancestor is closer than its own ancestors and its other
    descendants, by fewer edges where any are at the same time.
    """
    if isinstance(candidate_mask, str):
        candidate_mask = pl.col(candidate_mask)
    if isinstance(extant_mask, str):
        extant_mask = pl.col(extant_mask)
    if isinstance(criterion_cherry, str):
        criterion_cherry = pl.col(criterion_cherry)
    if isinstance(criterion_trunk, str):
        criterion_trunk = pl.col(criterion_trunk)
    if isinstance(criterion_tiebreak, str):
        criterion_tiebreak = pl.col(criterion_tiebreak)
    criteria = {
        "criterion_cherry": criterion_cherry,
        "criterion_trunk": criterion_trunk,
        "criterion_tiebreak": criterion_tiebreak,
    }

    phylogeny_df = alifestd_try_add_ancestor_id_col_polars(phylogeny_df)
    schema_names = phylogeny_df.lazy().collect_schema().names()
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

    selected = phylogeny_df
    if "node_depth" not in schema_names:
        selected = alifestd_mark_node_depth_polars(selected)
        schema_names = [*schema_names, "node_depth"]
    for name, criterion in criteria.items():
        for column in criterion.meta.root_names():
            if column not in schema_names:
                raise ValueError(
                    f"{name} column {column!r} not found in phylogeny_df",
                )

    candidate_rank = pl.struct(
        cherry=criterion_cherry, tiebreak=criterion_tiebreak
    ).rank("dense")
    selected = (
        selected.lazy()
        .select(
            pl.col("ancestor_id"),
            candidate_mask=candidate_mask.cast(pl.Boolean),
            extant_mask=extant_mask.cast(pl.Boolean),
            candidate_rank=(candidate_rank - 1).cast(pl.Int64),
            **criteria,
        )
        .collect()
    )
    for name in ("candidate_mask", "extant_mask"):
        if selected[name].has_nulls():
            raise NotImplementedError(f"null {name} values not supported")
    ancestor_ids = selected["ancestor_id"].to_numpy()
    # a row per taxon and a column per struct field, with nulls as nan
    values = {
        name: selected[name].to_numpy().reshape(len(selected), -1)
        for name in criteria
    }
    for name in criteria:
        if values[name].dtype.kind not in "iuf":
            raise NotImplementedError(
                f"non-numeric {name} values not supported",
            )
        if not np.isfinite(values[name]).all():
            raise NotImplementedError(
                f"null or non-finite {name} values not supported",
            )
    for name in ("criterion_cherry", "criterion_trunk"):
        if (values[name] < values[name][ancestor_ids]).any():
            raise NotImplementedError(
                f"{name} values decreasing from ancestor to descendant "
                "not supported",
            )
    # zero pad, so fields beyond the other criterion's are not counted for it
    cherry_values, trunk_values, tiebreak_values = values.values()
    num_fields = max(cherry_values.shape[1], trunk_values.shape[1])
    cherry_values, trunk_values = (
        np.pad(v, [(0, 0), (0, num_fields - v.shape[1])])
        for v in (cherry_values, trunk_values)
    )

    result = _alifestd_mask_persistence_fast_path(
        ancestor_ids,
        selected["candidate_mask"].to_numpy(),
        selected["extant_mask"].to_numpy(),
        selected["candidate_rank"].to_numpy(),
        cherry_values,
        trunk_values.max(axis=0) - trunk_values,
        tiebreak_values,
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

- Uses `--criterion-cherry` and `--criterion-trunk` to measure distance
from the most recent common ancestor of an extant taxon and a candidate out
to the candidate (cherry) and to the extant taxon (trunk). By default,
compares elapsed time by 'origin_time', then number of edges by
'node_depth', which is calculated if not present.

- Pass `--criterion-cherry 'pl.lit(0)'` to mark candidates by most recent
common ancestor, whatever their own branch lengths, or
`--criterion-trunk 'pl.lit(0)'` to mark candidates by how far they sit from
an extant taxon's lineage.

- Among candidates tied as an extant taxon's closest, the one with the
lowest `--criterion-tiebreak` value is marked (default: latest
'origin_time', then lowest 'id'). Pass `--criterion-tiebreak 'pl.lit(0)'`
to mark all tied candidates.

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
        dfcli_module="phyloframe.legacy._alifestd_mask_persistence_polars",
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
        default='pl.struct("origin_time", "node_depth")',
        type=str,
        help=(
            "Python expression evaluating to a column name or polars "
            "expression used to measure distance from MRCA to candidate, "
            "with polars available as `pl` "
            '(default: pl.struct("origin_time", "node_depth")).'
        ),
    )
    parser.add_argument(
        "--criterion-trunk",
        default='pl.struct("origin_time", "node_depth")',
        type=str,
        help=(
            "Python expression evaluating to a column name or polars "
            "expression used to measure distance from MRCA to extant taxon, "
            "with polars available as `pl` "
            '(default: pl.struct("origin_time", "node_depth")).'
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
    parser.add_argument(
        "--criterion-tiebreak",
        default=(
            'pl.struct(pl.col("origin_time").rank("dense", descending=True), '
            '"id").rank("ordinal")'
        ),
        type=str,
        help=(
            "Python expression evaluating to a column name or polars "
            "expression used to choose among tied candidates, taking the "
            "lowest, with polars available as `pl` "
            "(default: latest origin_time, then lowest id)."
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
            "criterion_trunk",
            "criterion_tiebreak",
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
