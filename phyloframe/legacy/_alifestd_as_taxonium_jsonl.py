import json
import logging
import typing

import numpy as np
import pandas as pd

from ._alifestd_has_contiguous_ids import alifestd_has_contiguous_ids
from ._alifestd_mark_num_leaves_asexual import (
    alifestd_mark_num_leaves_asexual,
)
from ._alifestd_mark_origin_time_delta_asexual import (
    alifestd_mark_origin_time_delta_asexual,
)
from ._alifestd_try_add_ancestor_id_col import alifestd_try_add_ancestor_id_col
from ._alifestd_unfurl_traversal_postorder_asexual import (
    alifestd_unfurl_traversal_postorder_asexual,
)


def _set_x_coords(
    ancestor_ids: np.ndarray,
    origin_time_deltas: np.ndarray,
) -> np.ndarray:
    """Compute cumulative distance from root for each node.

    Implementation detail for `alifestd_as_taxonium_jsonl`.
    """
    n = len(ancestor_ids)
    x_dist = np.zeros(n, dtype=np.float64)
    for i in range(n):
        aid = ancestor_ids[i]
        if aid != i:
            bl = origin_time_deltas[i]
            x_dist[i] = x_dist[aid] + (bl if not np.isnan(bl) else 0.0)
    return x_dist


def _set_y_coords(
    ancestor_ids: np.ndarray,
    postorder: np.ndarray,
) -> np.ndarray:
    """Compute y coordinates: leaves get sequential integers, internal
    nodes get the midpoint of their children's y range.

    Implementation detail for `alifestd_as_taxonium_jsonl`.
    """
    n = len(ancestor_ids)
    y = np.full(n, np.nan, dtype=np.float64)

    # identify leaves (nodes that are nobody's ancestor, except self-loops)
    has_child = np.zeros(n, dtype=np.bool_)
    for i in range(n):
        aid = ancestor_ids[i]
        if aid != i:
            has_child[aid] = True
    is_leaf = ~has_child

    # assign sequential y to leaves in preorder (postorder reversed)
    leaf_y = 0
    for node_id in postorder[::-1]:
        if is_leaf[node_id]:
            y[node_id] = leaf_y
            leaf_y += 1

    # assign internal nodes in postorder: midpoint of child y range
    child_min_y = np.full(n, np.inf, dtype=np.float64)
    child_max_y = np.full(n, -np.inf, dtype=np.float64)
    for node_id in postorder:
        aid = ancestor_ids[node_id]
        if aid != node_id:
            child_min_y[aid] = min(child_min_y[aid], y[node_id])
            child_max_y[aid] = max(child_max_y[aid], y[node_id])

    for node_id in postorder:
        if not is_leaf[node_id] and np.isfinite(child_min_y[node_id]):
            y[node_id] = (child_min_y[node_id] + child_max_y[node_id]) / 2.0

    return y


def alifestd_as_taxonium_jsonl(
    phylogeny_df: pd.DataFrame,
    mutate: bool = False,
    *,
    taxon_label: typing.Optional[str] = None,
    progress_wrap: typing.Callable = lambda x: x,
) -> str:
    """Convert phylogeny dataframe to Taxonium JSONL format.

    Produces a JSONL string where the first line contains tree-level
    metadata and each subsequent line represents a node with pre-computed
    layout coordinates, parent reference, and metadata.

    Parameters
    ----------
    phylogeny_df : pd.DataFrame
        Phylogeny dataframe in alife standard format.
    mutate : bool, optional
        Allow in-place mutations of the input dataframe, by default False.
    taxon_label : str, optional
        Column to use for node names, by default None (uses ``taxon_label``
        column if present, otherwise empty string).
    progress_wrap : typing.Callable, optional
        Pass tqdm or equivalent to display a progress bar.

    Returns
    -------
    str
        Taxonium JSONL formatted string.

    See Also
    --------
    alifestd_from_taxonium_jsonl :
        Inverse conversion, from Taxonium JSONL to alife standard format.
    """
    logging.info(
        "creating taxonium jsonl for alifestd df "
        f"with shape {phylogeny_df.shape}",
    )

    if not mutate:
        phylogeny_df = phylogeny_df.copy()

    phylogeny_df = alifestd_try_add_ancestor_id_col(phylogeny_df, mutate=True)

    if alifestd_has_contiguous_ids(phylogeny_df):
        phylogeny_df.reset_index(drop=True, inplace=True)
    else:
        phylogeny_df.index = phylogeny_df["id"]

    # set up origin_time_delta for branch lengths / x coordinates
    logging.info("setting up `origin_time_delta` column...")
    if "origin_time_delta" in phylogeny_df.columns:
        logging.info("... already present!")
    elif "origin_time" in phylogeny_df.columns:
        logging.info("... calculating from `origin_time`...")
        phylogeny_df = alifestd_mark_origin_time_delta_asexual(
            phylogeny_df, mutate=True
        )
    else:
        logging.info("... marking null")
        phylogeny_df["origin_time_delta"] = np.nan

    # mark num_leaves
    logging.info("marking num_leaves...")
    phylogeny_df = alifestd_mark_num_leaves_asexual(phylogeny_df, mutate=True)

    # compute postorder traversal
    logging.info("computing postorder traversal...")
    postorder_ids = alifestd_unfurl_traversal_postorder_asexual(phylogeny_df)

    # prepare arrays using contiguous id mapping
    ids = phylogeny_df["id"].values
    ancestor_ids_raw = phylogeny_df["ancestor_id"].values

    # build contiguous id mapping
    id_to_idx = {int(id_): i for i, id_ in enumerate(ids)}
    n = len(ids)
    ancestor_idx = np.array(
        [id_to_idx[int(aid)] for aid in ancestor_ids_raw], dtype=np.int64
    )
    postorder_idx = np.array(
        [id_to_idx[int(pid)] for pid in postorder_ids], dtype=np.int64
    )

    origin_time_deltas = phylogeny_df["origin_time_delta"].values.astype(
        np.float64
    )

    # compute layout coordinates
    logging.info("computing x coordinates...")
    x_dist = _set_x_coords(ancestor_idx, origin_time_deltas)

    # normalize x coordinates: 95th percentile maps to 600
    if n > 0 and np.any(x_dist > 0):
        sorted_x = np.sort(x_dist)
        p95 = sorted_x[int(len(sorted_x) * 0.95)]
        if p95 > 0:
            x_dist = 600.0 * (x_dist / p95)

    logging.info("computing y coordinates...")
    y_coords = _set_y_coords(ancestor_idx, postorder_idx)

    # prepare labels
    if taxon_label is not None:
        labels = phylogeny_df[taxon_label].astype(str).values
    elif "taxon_label" in phylogeny_df.columns:
        labels = phylogeny_df["taxon_label"].fillna("").astype(str).values
    else:
        labels = np.full(n, "", dtype=object)

    num_leaves = phylogeny_df["num_leaves"].values

    # identify leaf nodes
    has_child = np.zeros(n, dtype=np.bool_)
    for i in range(n):
        if ancestor_idx[i] != i:
            has_child[ancestor_idx[i]] = True
    is_tip = ~has_child

    # sort by y coordinate for taxonium output ordering
    sorted_indices = np.argsort(y_coords)
    idx_to_sorted = np.empty(n, dtype=np.int64)
    for sorted_pos, orig_idx in enumerate(sorted_indices):
        idx_to_sorted[orig_idx] = sorted_pos

    # count total tips
    total_tips = int(np.sum(is_tip))

    # build header line
    header = {
        "version": "0.0.0",
        "mutations": [],
        "total_nodes": n,
        "config": {"num_tips": total_tips},
    }

    lines = [json.dumps(header)]

    # build node lines in y-sorted order
    for sorted_pos in progress_wrap(range(n)):
        orig_idx = int(sorted_indices[sorted_pos])
        aid = ancestor_idx[orig_idx]
        node_obj = {
            "name": str(labels[orig_idx]),
            "x_dist": round(float(x_dist[orig_idx]), 5),
            "y": float(y_coords[orig_idx]),
            "mutations": [],
            "is_tip": bool(is_tip[orig_idx]),
            "parent_id": int(idx_to_sorted[aid]),
            "node_id": int(sorted_pos),
            "num_tips": int(num_leaves[orig_idx]),
        }
        lines.append(json.dumps(node_obj))

    logging.info(f"wrote {n} nodes to taxonium jsonl")
    return "\n".join(lines) + "\n"
