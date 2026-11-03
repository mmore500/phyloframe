import argparse
import functools
import heapq
import logging
import os
import typing

import joinem
from joinem._dataframe_cli import _add_parser_base, _run_dataframe_cli
import numpy as np
import pandas as pd

from .._auxlib._begin_prod_logging import begin_prod_logging
from .._auxlib._delegate_polars_implementation import (
    delegate_polars_implementation,
)
from .._auxlib._format_cli_description import format_cli_description
from .._auxlib._get_phyloframe_version import get_phyloframe_version
from .._auxlib._jit import jit
from .._auxlib._log_context_duration import log_context_duration
from .._auxlib._union_find_lookup_path_halving import (
    union_find_lookup_path_halving,
)
from ._alifestd_has_contiguous_ids import alifestd_has_contiguous_ids
from ._alifestd_is_topologically_sorted import alifestd_is_topologically_sorted
from ._alifestd_mark_leaves import alifestd_mark_leaves
from ._alifestd_try_add_ancestor_id_col import alifestd_try_add_ancestor_id_col


@jit(nopython=True)
def _alifestd_pool_chronological_consistency_asexual_fast_path(
    ancestor_ids: np.ndarray,
    origin_times: np.ndarray,
    weights: np.ndarray,
    is_pinned: np.ndarray,
) -> np.ndarray:
    """Implementation detail for
    `alifestd_pool_chronological_consistency_asexual` and
    `alifestd_pool_chronological_consistency_polars`.

    Requires contiguous ids and topologically sorted rows, float64
    `origin_times` and `weights`, and bool `is_pinned`. Returns pooled copy
    of `origin_times`.

    Taxa with positive weight that are not pinned ("active" taxa) undergo
    weighted isotonic regression over the forest induced by linking each to
    its nearest active ancestor not separated from it by a pinned taxon, via
    Thompson's minimum-block pooling algorithm: repeatedly, the unfinalized
    block with smallest mean is either finalized, if its topmost taxon has
    no unfinalized ancestor block, or pooled into its ancestor block. Pinned
    taxa keep their origin times, and active taxa are clamped between their
    nearest pinned ancestor and any descendant pinned taxa. As these bounds
    are themselves chronologically consistent (if pinned taxa are), such
    clamping preserves optimality. Remaining (zero-weight) taxa are clamped
    between their parent's and their descendants' result origin times.
    """
    num_taxa = len(ancestor_ids)
    is_active = (weights > 0.0) & ~is_pinned

    # link each taxon to nearest active proper ancestor, or -1 if none or if
    # a pinned taxon intervenes; pinned taxa partition pooling
    nearest_active = np.full(num_taxa, -1, dtype=np.int64)  # inclusive
    effective_parents = np.full(num_taxa, -1, dtype=np.int64)
    for idx, ancestor_id in enumerate(ancestor_ids):
        if ancestor_id != idx:  # exclude genesis cases
            effective_parents[idx] = nearest_active[ancestor_id]
        if is_active[idx]:
            nearest_active[idx] = idx
        elif is_pinned[idx]:
            nearest_active[idx] = -1
        else:
            nearest_active[idx] = effective_parents[idx]

    # blocks of pooled taxa, as union-find forest; topmost taxon of each block
    # is its union-find root, as blocks are only ever pooled into ancestors
    blocks = np.arange(num_taxa, dtype=np.int64)
    weight_sums = weights.copy()
    weighted_time_sums = weights * origin_times
    means = origin_times.copy()
    is_finalized = np.zeros(num_taxa, dtype=np.bool_)

    # min-heap of unfinalized blocks, keyed by mean then topmost taxon
    heap = [(means[idx], idx) for idx in range(num_taxa) if is_active[idx]]
    heapq.heapify(heap)
    while heap:
        mean, block = heapq.heappop(heap)
        if (
            blocks[block] != block
            or is_finalized[block]
            or means[block] != mean
        ):
            continue  # block since pooled away, finalized, or updated

        effective_parent = effective_parents[block]
        if effective_parent == -1:
            is_finalized[block] = True
            continue

        parent_block = union_find_lookup_path_halving(
            blocks,
            effective_parent,
        )
        if is_finalized[parent_block]:
            is_finalized[block] = True
            continue

        # parent block's mean is at least block's, as block has min mean
        blocks[block] = parent_block
        weight_sums[parent_block] += weight_sums[block]
        weighted_time_sums[parent_block] += weighted_time_sums[block]
        if means[block] != means[parent_block]:  # keep exact if tied
            means[parent_block] = (
                weighted_time_sums[parent_block] / weight_sums[parent_block]
            )
            # otherwise, mean is unchanged, so existing entry remains valid
            heapq.heappush(heap, (means[parent_block], parent_block))

    # lower bound on each taxon from ancestor pinned taxa, inclusive
    pinned_lower_bounds = np.where(is_pinned, origin_times, -np.inf)
    for idx, ancestor_id in enumerate(ancestor_ids):
        if ancestor_id != idx and not is_pinned[idx]:  # exclude genesis
            pinned_lower_bounds[idx] = pinned_lower_bounds[ancestor_id]

    # upper bound on each taxon from descendant pinned taxa, inclusive
    pinned_upper_bounds = np.where(is_pinned, origin_times, np.inf)
    for idx in range(num_taxa - 1, -1, -1):
        ancestor_id = ancestor_ids[idx]
        if ancestor_id != idx:  # exclude genesis cases
            pinned_upper_bounds[ancestor_id] = min(
                pinned_upper_bounds[ancestor_id],
                pinned_upper_bounds[idx],
            )

    pooled_times = origin_times.copy()
    for idx in range(num_taxa):
        if is_active[idx]:
            block = union_find_lookup_path_halving(blocks, idx)
            pooled_times[idx] = means[block]

    result = np.where(
        is_active,
        np.clip(pooled_times, pinned_lower_bounds, pinned_upper_bounds),
        origin_times,
    )

    # upper bound on each taxon from descendant active or pinned taxa
    is_settled = is_active | is_pinned
    settled_bounds = np.where(is_settled, result, np.inf)
    for idx in range(num_taxa - 1, -1, -1):
        ancestor_id = ancestor_ids[idx]
        if ancestor_id != idx:  # exclude genesis cases
            settled_bounds[ancestor_id] = min(
                settled_bounds[ancestor_id],
                settled_bounds[idx],
            )

    # clamp remaining taxa between parent and descendant settled taxa; as
    # bounds are consistent, parent result is max of its own origin time
    # (capped by its upper bound) and its lower bound, if not settled
    capped_times = np.where(
        is_settled, result, np.minimum(result, settled_bounds)
    )
    lower_bounds = np.full(num_taxa, -np.inf)
    for idx, ancestor_id in enumerate(ancestor_ids):
        if ancestor_id == idx:  # genesis case
            continue
        elif is_settled[ancestor_id]:
            lower_bounds[idx] = result[ancestor_id]
        else:
            lower_bounds[idx] = max(
                lower_bounds[ancestor_id],
                capped_times[ancestor_id],
            )

    return np.where(
        is_settled,
        result,
        np.clip(result, lower_bounds, settled_bounds),
    )


def _alifestd_pool_chronological_consistency_asexual_impl(
    ancestor_ids: np.ndarray,
    origin_times: np.ndarray,
    weights: typing.Optional[np.ndarray],
    is_pinned: typing.Optional[np.ndarray],
) -> np.ndarray:
    """Implementation detail for
    `alifestd_pool_chronological_consistency_asexual` and
    `alifestd_pool_chronological_consistency_polars`.

    Validates weights (uniform if None), pin mask (none pinned if None), and
    origin times, then dispatches to
    jit-compiled fast path. Returns pooled origin times, with dtype of
    `origin_times` if floating point and float64 otherwise.
    """
    ancestor_ids = np.asarray(ancestor_ids, dtype=np.int64)
    num_taxa = len(ancestor_ids)

    try:
        result_dtype = np.result_type(origin_times.dtype, np.float16)
        if not np.issubdtype(result_dtype, np.floating):
            result_dtype = np.float64
        origin_times = np.asarray(origin_times, dtype=np.float64)
    except (TypeError, ValueError):
        raise NotImplementedError("non-numeric origin_time not supported")
    if np.isnan(origin_times).any():
        raise ValueError("origin_time must not contain NaN values")

    if weights is None:
        weights = np.ones(num_taxa, dtype=np.float64)

    weights = np.asarray(weights, dtype=np.float64)
    if not (np.isfinite(weights) & (weights >= 0)).all():
        raise ValueError("weights must be finite and non-negative")

    if is_pinned is None:
        is_pinned = np.zeros(num_taxa, dtype=np.bool_)
    elif not np.issubdtype(np.asarray(is_pinned).dtype, np.bool_):
        raise ValueError("pin_by column must be boolean")
    is_pinned = np.asarray(is_pinned, dtype=np.bool_)

    result = _alifestd_pool_chronological_consistency_asexual_fast_path(
        ancestor_ids,
        origin_times,
        weights,
        is_pinned,
    )
    return result.astype(result_dtype, copy=False)


def alifestd_pool_chronological_consistency_asexual(
    phylogeny_df: pd.DataFrame,
    mutate: bool = False,
    *,
    weight_by: typing.Optional[str] = None,
    pin_by: typing.Optional[str] = "is_leaf",
) -> pd.DataFrame:
    """Resolve origin time inversions, without changing topology, by pooling
    inverted taxa with their ancestors, setting each pooled block to its
    weighted mean origin time.

    Performs weighted isotonic regression of origin times over the tree,
    i.e., finds origin times that minimize
    `sum_v w_v * (t'_v - t_v) ** 2` while ensuring that no taxon precedes
    its parent.
    Unlike `alifestd_coerce_chronological_consistency`, which only ever
    raises origin times, pooling is unbiased: the weighted sum of origin
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

    Input dataframe is not mutated by this operation unless `mutate` set True.
    If mutate set True, operation does not occur in place; still use return
    value to get transformed phylogeny dataframe.

    Parameters
    ----------
    phylogeny_df : pandas.DataFrame
        The phylogeny as a dataframe in alife standard format.

        Must represent an asexual phylogeny with contiguous ids, numeric
        `origin_time`, and topologically sorted rows.
    mutate : bool, default False
        Allow in-place mutations of the input dataframe.
    weight_by : str, optional
        Name of column holding per-taxon pooling weights, which must be
        finite and non-negative. If None (default), all taxa have unit weight,
        minimizing unweighted squared origin time change.

        To weight each taxon by the number of leaf pairs with it as most
        recent common ancestor, conserving mean MRCA origin time over all
        leaf pairs, add a `num_mrca_leaf_pairs` column with
        `alifestd_mark_num_mrca_leaf_pairs_asexual` and pass
        ``weight_by="num_mrca_leaf_pairs"``.
    pin_by : str, optional, default "is_leaf"
        Name of boolean column marking taxa whose origin times are kept
        unchanged, excluding them from pooling. If None, no taxa are pinned.

        By default, leaves are pinned. If the `is_leaf` column is absent, it
        is computed with `alifestd_mark_leaves` (and not retained in output).

    Returns
    -------
    pandas.DataFrame
        The phylogeny with pooled `origin_time` column. Column dtype is
        float64, unless input dtype is another floating point type.

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
    alifestd_pool_chronological_consistency_polars :
        Polars-based implementation.
    alifestd_coerce_chronological_consistency :
        Resolves origin time inversions by raising origin times.
    alifestd_graft_chronological_consistency_asexual :
        Resolves origin time inversions by grafting onto nearest
        eligible ancestor node.
    alifestd_splice_chronological_consistency_asexual :
        Resolves origin time inversions by grafting onto a node spliced
        into nearest eligible ancestor edge.
    alifestd_find_chronological_inconsistency :
        Finds taxa with origin time inversions.
    alifestd_mark_num_mrca_leaf_pairs_asexual :
        Computes `num_mrca_leaf_pairs` weights.
    """
    if not mutate:
        phylogeny_df = phylogeny_df.copy()

    phylogeny_df = alifestd_try_add_ancestor_id_col(phylogeny_df, mutate=True)
    if "ancestor_id" not in phylogeny_df.columns:
        raise NotImplementedError(
            "alifestd_pool_chronological_consistency_asexual only "
            "supports asexual phylogenies.",
        )

    if phylogeny_df.empty:
        return phylogeny_df

    if not alifestd_has_contiguous_ids(phylogeny_df):
        raise NotImplementedError("non-contiguous ids not supported")

    if not alifestd_is_topologically_sorted(phylogeny_df):
        raise NotImplementedError(
            "non-topologically-sorted data not supported",
        )

    had_is_leaf = "is_leaf" in phylogeny_df.columns
    if pin_by == "is_leaf" and not had_is_leaf:
        phylogeny_df = alifestd_mark_leaves(phylogeny_df, mutate=True)

    phylogeny_df["origin_time"] = (
        _alifestd_pool_chronological_consistency_asexual_impl(
            phylogeny_df["ancestor_id"].to_numpy(),
            phylogeny_df["origin_time"].to_numpy(),
            None if weight_by is None else phylogeny_df[weight_by].to_numpy(),
            None if pin_by is None else phylogeny_df[pin_by].to_numpy(),
        )
    )

    if not had_is_leaf and "is_leaf" in phylogeny_df.columns:
        phylogeny_df.drop(columns=["is_leaf"], inplace=True)

    return phylogeny_df


_raw_description = f"""{os.path.basename(__file__)} | (phyloframe v{get_phyloframe_version()}/joinem v{joinem.__version__})

Resolve origin time inversions by pooling inverted taxa with their ancestors, setting each pooled block to its weighted mean origin time (i.e., weighted isotonic regression over the tree).

Use `--weight-by` to weight taxa by a column (e.g., `num_mrca_leaf_pairs`, as added by `alifestd_mark_num_mrca_leaf_pairs_asexual`); otherwise, taxa are uniformly weighted.

Use `--pin-by` to keep origin times of taxa marked by a boolean column unchanged (default: `is_leaf`, computed if absent); pass `--no-pin` to pin no taxa.

Data is assumed to be in alife standard format.

Additional Notes
================
- Requires asexual phylogeny with contiguous ids and topologically sorted rows.

- Use `--eager-read` if modifying data file inplace.

- This CLI entrypoint is experimental and may be subject to change.

See Also
========
phyloframe.legacy._alifestd_pool_chronological_consistency_polars :
    Entrypoint for high-performance Polars-based implementation.
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
        dfcli_module="phyloframe.legacy._alifestd_pool_chronological_consistency_asexual",
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
    with log_context_duration(
        "phyloframe.legacy._alifestd_pool_chronological_consistency_asexual",
        logging.info,
    ):
        _run_dataframe_cli(
            base_parser=parser,
            output_dataframe_op=delegate_polars_implementation()(
                functools.partial(
                    alifestd_pool_chronological_consistency_asexual,
                    weight_by=args.weight_by,
                    pin_by=None if args.no_pin else args.pin_by,
                ),
            ),
        )
