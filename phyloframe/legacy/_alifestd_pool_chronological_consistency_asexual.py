import argparse
import functools
import logging
import os
import typing

import joinem
from joinem._dataframe_cli import _add_parser_base, _run_dataframe_cli
import numpy as np
import pandas as pd

from .._auxlib._add_bool_arg import add_bool_arg
from .._auxlib._begin_prod_logging import begin_prod_logging
from .._auxlib._delegate_polars_implementation import (
    delegate_polars_implementation,
)
from .._auxlib._format_cli_description import format_cli_description
from .._auxlib._get_phyloframe_version import get_phyloframe_version
from .._auxlib._jit import jit
from .._auxlib._log_context_duration import log_context_duration
from ._alifestd_has_contiguous_ids import alifestd_has_contiguous_ids
from ._alifestd_is_topologically_sorted import alifestd_is_topologically_sorted
from ._alifestd_mark_num_mrca_leaf_pairs_asexual import (
    _alifestd_mark_num_mrca_leaf_pairs_asexual_fast_path,
)
from ._alifestd_try_add_ancestor_id_col import alifestd_try_add_ancestor_id_col


@jit(nopython=True)
def _alifestd_pool_chronological_consistency_heap_push(
    heap_keys: np.ndarray,
    heap_nodes: np.ndarray,
    heap_versions: np.ndarray,
    size: int,
    key: float,
    node: int,
    version: int,
) -> int:
    """Implementation detail for
    `_alifestd_pool_chronological_consistency_asexual_fast_path`.

    Pushes onto binary min-heap ordered by key, then node, stored across
    parallel arrays. Returns new heap size.
    """
    pos = size
    while pos > 0:
        parent = (pos - 1) // 2
        if heap_keys[parent] < key or (
            heap_keys[parent] == key and heap_nodes[parent] <= node
        ):
            break
        heap_keys[pos] = heap_keys[parent]
        heap_nodes[pos] = heap_nodes[parent]
        heap_versions[pos] = heap_versions[parent]
        pos = parent

    heap_keys[pos] = key
    heap_nodes[pos] = node
    heap_versions[pos] = version
    return size + 1


@jit(nopython=True)
def _alifestd_pool_chronological_consistency_heap_pop(
    heap_keys: np.ndarray,
    heap_nodes: np.ndarray,
    heap_versions: np.ndarray,
    size: int,
) -> typing.Tuple[int, int, int]:
    """Implementation detail for
    `_alifestd_pool_chronological_consistency_asexual_fast_path`.

    Pops minimum from binary min-heap stored across parallel arrays. Returns
    popped node, its version, and new heap size.
    """
    node = heap_nodes[0]
    version = heap_versions[0]

    size -= 1
    key = heap_keys[size]
    last_node = heap_nodes[size]
    last_version = heap_versions[size]

    pos = 0
    while True:
        child = 2 * pos + 1
        if child >= size:
            break
        if child + 1 < size and (
            heap_keys[child + 1] < heap_keys[child]
            or (
                heap_keys[child + 1] == heap_keys[child]
                and heap_nodes[child + 1] < heap_nodes[child]
            )
        ):
            child += 1
        if key < heap_keys[child] or (
            key == heap_keys[child] and last_node <= heap_nodes[child]
        ):
            break
        heap_keys[pos] = heap_keys[child]
        heap_nodes[pos] = heap_nodes[child]
        heap_versions[pos] = heap_versions[child]
        pos = child

    if size > 0:
        heap_keys[pos] = key
        heap_nodes[pos] = last_node
        heap_versions[pos] = last_version

    return node, version, size


@jit(nopython=True)
def _alifestd_pool_chronological_consistency_find(
    blocks: np.ndarray,
    node: int,
) -> int:
    """Implementation detail for
    `_alifestd_pool_chronological_consistency_asexual_fast_path`.

    Union-find lookup with path halving.
    """
    while blocks[node] != node:
        blocks[node] = blocks[blocks[node]]
        node = blocks[node]
    return node


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
    its nearest active ancestor, via Thompson's minimum-block pooling
    algorithm: repeatedly, the unfinalized block with smallest mean is
    either finalized, if its topmost taxon has no unfinalized ancestor
    block, or pooled into its ancestor block. Pinned taxa keep their origin
    times, and active taxa are clamped to not exceed any descendant pinned
    taxon. Remaining (zero-weight) taxa are clamped between their parent's
    and their descendants' result origin times.
    """
    num_taxa = len(ancestor_ids)
    is_active = (weights > 0.0) & ~is_pinned

    # link each taxon to nearest active proper ancestor, or -1 if none
    nearest_active = np.full(num_taxa, -1, dtype=np.int64)  # inclusive
    effective_parents = np.full(num_taxa, -1, dtype=np.int64)
    for idx, ancestor_id in enumerate(ancestor_ids):
        if ancestor_id != idx:  # exclude genesis cases
            effective_parents[idx] = nearest_active[ancestor_id]
        if is_active[idx]:
            nearest_active[idx] = idx
        else:
            nearest_active[idx] = effective_parents[idx]

    # blocks of pooled taxa, as union-find forest; topmost taxon of each block
    # is its union-find root, as blocks are only ever pooled into ancestors
    blocks = np.arange(num_taxa, dtype=np.int64)
    weight_sums = weights.copy()
    weighted_time_sums = weights * origin_times
    means = origin_times.copy()
    versions = np.zeros(num_taxa, dtype=np.int64)  # invalidates heap entries
    is_finalized = np.zeros(num_taxa, dtype=np.bool_)

    # each pooling pushes one entry, so at most 2 * num_taxa pushes
    heap_keys = np.empty(2 * num_taxa, dtype=np.float64)
    heap_nodes = np.empty(2 * num_taxa, dtype=np.int64)
    heap_versions = np.empty(2 * num_taxa, dtype=np.int64)
    heap_size = 0
    for idx in range(num_taxa):
        if is_active[idx]:
            heap_size = _alifestd_pool_chronological_consistency_heap_push(
                heap_keys,
                heap_nodes,
                heap_versions,
                heap_size,
                means[idx],
                idx,
                0,
            )

    while heap_size > 0:
        (
            block,
            version,
            heap_size,
        ) = _alifestd_pool_chronological_consistency_heap_pop(
            heap_keys,
            heap_nodes,
            heap_versions,
            heap_size,
        )
        if blocks[block] != block or versions[block] != version:
            continue  # stale entry, block pooled away or since updated

        effective_parent = effective_parents[block]
        if effective_parent == -1:
            is_finalized[block] = True
            continue

        parent_block = _alifestd_pool_chronological_consistency_find(
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
        versions[parent_block] += 1
        heap_size = _alifestd_pool_chronological_consistency_heap_push(
            heap_keys,
            heap_nodes,
            heap_versions,
            heap_size,
            means[parent_block],
            parent_block,
            versions[parent_block],
        )

    # upper bound on each taxon from descendant pinned taxa, inclusive
    pinned_bounds = np.where(is_pinned, origin_times, np.inf)
    for idx in range(num_taxa - 1, -1, -1):
        ancestor_id = ancestor_ids[idx]
        if ancestor_id != idx:  # exclude genesis cases
            pinned_bounds[ancestor_id] = min(
                pinned_bounds[ancestor_id],
                pinned_bounds[idx],
            )

    result = origin_times.copy()
    for idx in range(num_taxa):
        if is_active[idx]:
            block = _alifestd_pool_chronological_consistency_find(blocks, idx)
            result[idx] = min(means[block], pinned_bounds[idx])

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

    # clamp remaining taxa between parent and descendant settled taxa
    for idx, ancestor_id in enumerate(ancestor_ids):
        if is_settled[idx]:
            continue
        lower_bound = result[ancestor_id] if ancestor_id != idx else -np.inf
        result[idx] = min(max(result[idx], lower_bound), settled_bounds[idx])

    return result


def _alifestd_pool_chronological_consistency_asexual_impl(
    ancestor_ids: np.ndarray,
    origin_times: np.ndarray,
    weights: typing.Union[str, np.ndarray],
    pin_leaves: bool,
) -> np.ndarray:
    """Implementation detail for
    `alifestd_pool_chronological_consistency_asexual` and
    `alifestd_pool_chronological_consistency_polars`.

    Resolves and validates weights and origin times, then dispatches to
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

    if isinstance(weights, str) and weights == "mrca_leaf_pairs":
        weights = _alifestd_mark_num_mrca_leaf_pairs_asexual_fast_path(
            ancestor_ids,
        )
    elif isinstance(weights, str) and weights == "uniform":
        weights = np.ones(num_taxa, dtype=np.float64)
    elif isinstance(weights, str):
        raise ValueError(f"unrecognized weight_by {weights!r}")

    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != (num_taxa,):
        raise ValueError(
            f"weights must have one value per taxon, expected shape "
            f"{(num_taxa,)} but got {weights.shape}",
        )
    if not (np.isfinite(weights) & (weights >= 0)).all():
        raise ValueError("weights must be finite and non-negative")

    is_pinned = np.zeros(num_taxa, dtype=np.bool_)
    if pin_leaves:
        is_pinned[:] = True
        is_nonroot = ancestor_ids != np.arange(num_taxa)
        is_pinned[ancestor_ids[is_nonroot]] = False

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
    weight_by: typing.Union[
        typing.Literal["mrca_leaf_pairs", "uniform"],
        str,
        np.ndarray,
        pd.Series,
    ] = "mrca_leaf_pairs",
    pin_leaves: bool = True,
) -> pd.DataFrame:
    """Resolve origin time inversions by pooling inverted taxa with their
    ancestors, setting each pooled block to its weighted mean origin time.

    Performs weighted isotonic regression of origin times over the tree,
    i.e., finds origin times minimizing `sum_v w_v * (t'_v - t_v) ** 2`
    subject to no taxon preceding its parent. Unlike
    `alifestd_coerce_chronological_consistency`, which only ever raises
    origin times, pooling is unbiased: the weighted sum of origin times,
    `sum_v w_v * t_v`, is conserved exactly within each pooled block.
    Topology is unchanged.

    Taxa with zero weight do not participate in pooling. Instead, after
    pooling, each is clamped between its parent's origin time and the
    minimum origin time among its descendants with positive weight (or
    pinned). Under the default `mrca_leaf_pairs` weighting, this is the
    case for unifurcations.

    If `pin_leaves` is set, leaf origin times are kept unchanged, and leaves
    are excluded from pooling. Taxa with positive weight are then clamped
    not to exceed origin times of any descendant leaves; only such clamping,
    which is necessary only if a leaf precedes pooled ancestors, breaks
    conservation of weighted origin time sum.

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
    weight_by : str, numpy.ndarray, or pandas.Series, default "mrca_leaf_pairs"
        Per-taxon pooling weights, which must be finite and non-negative.

        - ``"mrca_leaf_pairs"``: number of leaf pairs with each taxon as
          most recent common ancestor (see
          `alifestd_mark_num_mrca_leaf_pairs_asexual`). Conserves mean MRCA
          origin time over all leaf pairs.
        - ``"uniform"``: unit weight for all taxa, minimizing unweighted
          squared origin time change.
        - other str: name of column holding weights.
        - array-like: weights, aligned with rows.
    pin_leaves : bool, default True
        Keep leaf origin times unchanged, excluding leaves from pooling.

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
        If weights are invalid or `origin_time` contains NaN values.

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
        Computes default `mrca_leaf_pairs` weights.
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

    if isinstance(weight_by, str) and weight_by not in (
        "mrca_leaf_pairs",
        "uniform",
    ):
        weight_by = phylogeny_df[weight_by].to_numpy()
    elif isinstance(weight_by, pd.Series):
        weight_by = weight_by.to_numpy()

    phylogeny_df[
        "origin_time"
    ] = _alifestd_pool_chronological_consistency_asexual_impl(
        phylogeny_df["ancestor_id"].to_numpy(),
        phylogeny_df["origin_time"].to_numpy(),
        weight_by,
        pin_leaves,
    )
    return phylogeny_df


_raw_description = f"""{os.path.basename(__file__)} | (phyloframe v{get_phyloframe_version()}/joinem v{joinem.__version__})

Resolve origin time inversions by pooling inverted taxa with their ancestors, setting each pooled block to its weighted mean origin time (i.e., weighted isotonic regression over the tree).

Use `--weight-by` to choose weights: `mrca_leaf_pairs` (default), `uniform`, or a column name.

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
        default="mrca_leaf_pairs",
        type=str,
        help=(
            "pooling weights: 'mrca_leaf_pairs', 'uniform', or column name "
            "(default: mrca_leaf_pairs)"
        ),
    )
    add_bool_arg(
        parser,
        "pin-leaves",
        default=True,
        help="keep leaf origin times unchanged (default: True)",
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
                    pin_leaves=args.pin_leaves,
                ),
            ),
        )
