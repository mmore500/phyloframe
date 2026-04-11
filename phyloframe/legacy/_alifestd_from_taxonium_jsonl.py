import json
import logging
import typing

import numpy as np
import pandas as pd

from ._alifestd_make_ancestor_list_col import alifestd_make_ancestor_list_col


def alifestd_from_taxonium_jsonl(
    taxonium_jsonl: str,
    *,
    create_ancestor_list: bool = False,
    dtype_id: typing.Optional[type] = np.int64,
) -> pd.DataFrame:
    """Convert a Taxonium JSONL string to a phylogeny dataframe.

    Parses a Taxonium JSONL string (where the first line is tree-level
    metadata and subsequent lines are node records) and returns a pandas
    DataFrame in alife standard format with columns: id, ancestor_id,
    taxon_label, origin_time_delta, and branch_length. Optionally includes
    ancestor_list.

    Parameters
    ----------
    taxonium_jsonl : str
        A phylogeny in Taxonium JSONL format.
    create_ancestor_list : bool, default False
        If True, include an ``ancestor_list`` column in the result.
    dtype_id : type or None, default np.int64
        Numpy dtype for the ``id`` and ``ancestor_id`` columns. If None, the
        smallest signed integer dtype is chosen automatically based on the
        number of nodes.

    Returns
    -------
    pd.DataFrame
        Phylogeny dataframe in alife standard format.

    See Also
    --------
    alifestd_as_taxonium_jsonl :
        Inverse conversion, from alife standard to Taxonium JSONL format.
    """
    lines = taxonium_jsonl.strip().splitlines()
    if not lines:
        resolved_dtype_id = np.dtype(dtype_id or np.int8)
        columns = {
            "id": pd.Series(dtype=resolved_dtype_id),
            "ancestor_id": pd.Series(dtype=resolved_dtype_id),
            "taxon_label": pd.Series(dtype=str),
            "origin_time_delta": pd.Series(dtype=float),
            "branch_length": pd.Series(dtype=float),
        }
        if create_ancestor_list:
            columns["ancestor_list"] = pd.Series(dtype=str)
        return pd.DataFrame(columns)

    # parse header (first line)
    header = json.loads(lines[0])
    logging.info(
        f"parsed taxonium jsonl header: {header.get('total_nodes', '?')} "
        "nodes expected",
    )

    # parse node lines
    node_lines = lines[1:]
    num_nodes = len(node_lines)

    if num_nodes == 0:
        resolved_dtype_id = np.dtype(dtype_id or np.int8)
        columns = {
            "id": pd.Series(dtype=resolved_dtype_id),
            "ancestor_id": pd.Series(dtype=resolved_dtype_id),
            "taxon_label": pd.Series(dtype=str),
            "origin_time_delta": pd.Series(dtype=float),
            "branch_length": pd.Series(dtype=float),
        }
        if create_ancestor_list:
            columns["ancestor_list"] = pd.Series(dtype=str)
        return pd.DataFrame(columns)

    if dtype_id is None:
        resolved_dtype_id = np.min_scalar_type(-max(num_nodes - 1, 1))
    else:
        resolved_dtype_id = np.dtype(dtype_id)

    node_ids = np.empty(num_nodes, dtype=np.int64)
    parent_ids = np.empty(num_nodes, dtype=np.int64)
    names = np.empty(num_nodes, dtype=object)
    x_dists = np.empty(num_nodes, dtype=np.float64)

    for i, line in enumerate(node_lines):
        node = json.loads(line)
        node_ids[i] = node["node_id"]
        parent_ids[i] = node["parent_id"]
        names[i] = node.get("name", "")
        x_dists[i] = node.get("x_dist", np.nan)

    # taxonium uses node_id as position in sorted order; map to alife
    # standard contiguous ids preserving the node_id ordering
    # build mapping from taxonium node_id -> alife id
    alife_ids = np.arange(num_nodes, dtype=resolved_dtype_id)

    # map parent_ids through the same mapping
    # taxonium node_ids should be 0..n-1, but handle sparse case
    taxonium_to_alife = np.empty(
        int(node_ids.max()) + 1, dtype=resolved_dtype_id
    )
    for i in range(num_nodes):
        taxonium_to_alife[node_ids[i]] = alife_ids[i]

    ancestor_alife_ids = np.array(
        [taxonium_to_alife[pid] for pid in parent_ids],
        dtype=resolved_dtype_id,
    )

    # reorder arrays so they are sorted by alife id (which is node_id order)
    sort_order = np.argsort(node_ids)
    alife_ids = alife_ids[sort_order]
    ancestor_alife_ids = ancestor_alife_ids[sort_order]
    names = names[sort_order]
    x_dists = x_dists[sort_order]

    # compute branch lengths from x_dist differences
    # branch_length[i] = x_dist[i] - x_dist[parent[i]]
    branch_lengths = np.full(num_nodes, np.nan, dtype=np.float64)
    for i in range(num_nodes):
        aid = int(ancestor_alife_ids[i])
        if aid != int(alife_ids[i]):  # not a root
            branch_lengths[i] = x_dists[i] - x_dists[aid]
        # root gets NaN branch length

    phylogeny_df = pd.DataFrame(
        {
            "id": alife_ids,
            "ancestor_id": ancestor_alife_ids,
            "taxon_label": names,
            "origin_time_delta": branch_lengths.copy(),
            "branch_length": branch_lengths,
        },
    )

    if create_ancestor_list:
        phylogeny_df["ancestor_list"] = alifestd_make_ancestor_list_col(
            phylogeny_df["id"],
            phylogeny_df["ancestor_id"],
        )

    logging.info(f"parsed {num_nodes} nodes from taxonium jsonl")
    return phylogeny_df
