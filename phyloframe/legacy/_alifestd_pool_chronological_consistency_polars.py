import argparse
import functools
import logging
import os
import typing

import joinem
from joinem._dataframe_cli import _add_parser_base, _run_dataframe_cli
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
from ._alifestd_mark_leaves_polars import alifestd_mark_leaves_polars
from ._alifestd_pool_chronological_consistency_asexual import (
    _alifestd_pool_chronological_consistency_asexual_impl,
)
from ._alifestd_try_add_ancestor_id_col_polars import (
    alifestd_try_add_ancestor_id_col_polars,
)


def alifestd_pool_chronological_consistency_polars(
    phylogeny_df: pl.DataFrame,
    *,
    weight_by: typing.Optional[str] = None,
    pin_by: typing.Optional[str] = "is_leaf",
) -> pl.DataFrame:
    """Resolve origin time inversions, without changing topology, by pooling
    inverted taxa with their ancestors, setting each pooled block to its
    weighted mean origin time.

    Performs weighted isotonic regression of origin times over the tree,
    i.e., finds origin times that minimize
    `sum_v w_v * (t'_v - t_v) ** 2` while ensuring that no taxon precedes
    its parent.
    Unlike `alifestd_coerce_chronological_consistency_polars`, which only
    ever raises origin times, pooling is unbiased: the weighted sum of origin
    times, `sum_v w_v * t_v`, is conserved exactly within each pooled block.

    Taxa with zero weight do not participate in pooling. Instead, after
    pooling, each is clamped between its parent's origin time and the
    minimum origin time among its descendants with positive weight (or
    pinned). Under `num_mrca_leaf_pairs` weighting (see `weight_by`), this
    is the case for leaves and unifurcations.

    Pinned taxa (see `pin_by`; by default, leaves) keep their origin times
    unchanged, and are excluded from pooling. Taxa with positive weight are
    then clamped between origin times of their nearest pinned ancestor and
    any descendant pinned taxa; only such clamping breaks conservation of
    weighted origin time sum. Pinned taxa should themselves be
    chronologically consistent, as their origin times are never changed;
    if a pinned taxon precedes a pinned ancestor, the result will retain
    that inversion.

    Parameters
    ----------
    phylogeny_df : polars.DataFrame or polars.LazyFrame
        The phylogeny as a dataframe in alife standard format.

        Must represent an asexual phylogeny with contiguous ids, numeric
        `origin_time`, and topologically sorted rows.
    weight_by : str, optional
        Name of column holding per-taxon pooling weights, which must be
        finite and non-negative. If None (default), all taxa have unit weight,
        minimizing unweighted squared origin time change.

        To weight each taxon by the number of leaf pairs with it as most
        recent common ancestor, conserving mean MRCA origin time over all
        leaf pairs, add a `num_mrca_leaf_pairs` column with
        `alifestd_mark_num_mrca_leaf_pairs_polars` and pass
        ``weight_by="num_mrca_leaf_pairs"``.
    pin_by : str, optional, default "is_leaf"
        Name of boolean column marking taxa whose origin times are kept
        unchanged, excluding them from pooling. If None, no taxa are pinned.

        By default, leaves are pinned. If the `is_leaf` column is absent, it
        is computed with `alifestd_mark_leaves_polars` (and not retained in
        output).

    Returns
    -------
    polars.DataFrame or polars.LazyFrame
        The phylogeny with pooled `origin_time` column. Column dtype is
        Float64, unless input dtype is another floating point type.

    Raises
    ------
    NotImplementedError
        If `phylogeny_df` is sexual, has non-contiguous ids, is not
        topologically sorted, or has non-numeric `origin_time`.
    ValueError
        If weights are invalid, `pin_by` column is not boolean, or
        `origin_time` contains NaN values.

    See Also
    --------
    alifestd_pool_chronological_consistency_asexual :
        Pandas-based implementation.
    alifestd_coerce_chronological_consistency_polars :
        Resolves origin time inversions by raising origin times.
    alifestd_graft_chronological_consistency_polars :
        Resolves origin time inversions by grafting onto nearest
        eligible ancestor node.
    alifestd_splice_chronological_consistency_polars :
        Resolves origin time inversions by grafting onto a node spliced
        into nearest eligible ancestor edge.
    alifestd_find_chronological_inconsistency_polars :
        Finds taxa with origin time inversions.
    alifestd_mark_num_mrca_leaf_pairs_polars :
        Computes `num_mrca_leaf_pairs` weights.
    """

    logging.info(
        "- alifestd_pool_chronological_consistency_polars: "
        "adding ancestor_id col...",
    )
    phylogeny_df = alifestd_try_add_ancestor_id_col_polars(phylogeny_df)

    schema = phylogeny_df.lazy().collect_schema()
    if "ancestor_id" not in schema.names():
        raise NotImplementedError("sexual phylogenies not supported")

    if phylogeny_df.lazy().limit(1).collect().is_empty():
        return phylogeny_df

    if not alifestd_has_contiguous_ids_polars(phylogeny_df):
        raise NotImplementedError("non-contiguous ids not supported")

    if not alifestd_is_topologically_sorted_polars(phylogeny_df):
        raise NotImplementedError(
            "non-topologically-sorted data not supported",
        )

    logging.info(
        "- alifestd_pool_chronological_consistency_polars: "
        "extracting columns...",
    )
    ancestor_ids = (
        phylogeny_df.lazy()
        .select("ancestor_id")
        .collect()
        .to_series()
        .to_numpy()
    )
    origin_times = (
        phylogeny_df.lazy()
        .select("origin_time")
        .collect()
        .to_series()
        .to_numpy()
    )
    weights = (
        None
        if weight_by is None
        else phylogeny_df.lazy()
        .select(weight_by)
        .collect()
        .to_series()
        .to_numpy()
    )

    if pin_by == "is_leaf" and "is_leaf" not in schema.names():
        logging.info(
            "- alifestd_pool_chronological_consistency_polars: "
            "marking leaves...",
        )
        is_pinned = (
            alifestd_mark_leaves_polars(phylogeny_df.lazy())
            .select("is_leaf")
            .collect()
            .to_series()
            .to_numpy()
        )
    elif pin_by is not None:
        is_pinned = (
            phylogeny_df.lazy().select(pin_by).collect().to_series().to_numpy()
        )
    else:
        is_pinned = None

    logging.info(
        "- alifestd_pool_chronological_consistency_polars: "
        "pooling origin times...",
    )
    pooled = _alifestd_pool_chronological_consistency_asexual_impl(
        ancestor_ids,
        origin_times,
        weights,
        is_pinned,
    )

    return phylogeny_df.with_columns(pl.Series("origin_time", pooled))


_raw_description = f"""{os.path.basename(__file__)} | (phyloframe v{get_phyloframe_version()}/joinem v{joinem.__version__})

Resolve origin time inversions by pooling inverted taxa with their ancestors, setting each pooled block to its weighted mean origin time (i.e., weighted isotonic regression over the tree).

Use `--weight-by` to weight taxa by a column (e.g., `num_mrca_leaf_pairs`, as added by `alifestd_mark_num_mrca_leaf_pairs_polars`); otherwise, taxa are uniformly weighted.

Use `--pin-by` to keep origin times of taxa marked by a boolean column unchanged (default: `is_leaf`, computed if absent); pass `--no-pin` to pin no taxa.

Data is assumed to be in alife standard format.

Additional Notes
================
- Requires asexual phylogeny with contiguous ids and topologically sorted rows.

- Use `--eager-read` if modifying data file inplace.

- This CLI entrypoint is experimental and may be subject to change.

See Also
========
phyloframe.legacy._alifestd_pool_chronological_consistency_asexual :
    CLI entrypoint for Pandas-based implementation.
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
        dfcli_module="phyloframe.legacy._alifestd_pool_chronological_consistency_polars",
        dfcli_version=get_phyloframe_version(),
    )
    parser.add_argument(
        "--weight-by",
        default=None,
        type=str,
        help="column name of pooling weights (default: None, i.e., uniform)",
    )
    parser.add_argument(
        "--pin-by",
        default="is_leaf",
        type=str,
        help=(
            "column name of boolean mask for taxa with origin times kept "
            "unchanged (default: is_leaf, computed if absent)"
        ),
    )
    parser.add_argument(
        "--no-pin",
        action="store_true",
        default=False,
        help="pin no taxa, overriding --pin-by",
    )
    return parser


if __name__ == "__main__":
    begin_prod_logging()

    parser = _create_parser()
    args, __ = parser.parse_known_args()

    try:
        with log_context_duration(
            "phyloframe.legacy._alifestd_pool_chronological_consistency_polars",
            logging.info,
        ):
            _run_dataframe_cli(
                base_parser=parser,
                output_dataframe_op=functools.partial(
                    alifestd_pool_chronological_consistency_polars,
                    weight_by=args.weight_by,
                    pin_by=None if args.no_pin else args.pin_by,
                ),
            )
    except NotImplementedError as e:
        logging.error(
            "- polars op not yet implemented, use pandas op CLI instead",
        )
        raise e
