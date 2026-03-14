import argparse
import functools
import logging
import os

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
from ._alifestd_is_topologically_sorted import (
    alifestd_is_topologically_sorted,
)
from ._alifestd_mark_first_child_id_asexual import (
    _alifestd_mark_first_child_id_asexual_fast_path,
)
from ._alifestd_mark_next_sibling_id_asexual import (
    _alifestd_mark_next_sibling_id_asexual_fast_path,
)
from ._alifestd_topological_sort import alifestd_topological_sort
from ._alifestd_try_add_ancestor_id_col import (
    alifestd_try_add_ancestor_id_col,
)


@jit(nopython=True)
def _alifestd_sort_children_asexual_fast_path(
    ancestor_ids: np.ndarray,
    criterion_values: np.ndarray,
    first_child_ids: np.ndarray,
    next_sibling_ids: np.ndarray,
    reverse: bool = False,
) -> np.ndarray:
    """Implementation detail for `alifestd_sort_children_asexual`.

    Returns a permutation array giving the row order after sorting children
    by criterion values and performing a preorder DFS.

    Uses first_child_id / next_sibling_id linked-list representation from
    PR #36 to traverse children of each node.

    Assumes contiguous ids and topological sorting.
    """
    n = len(ancestor_ids)

    # Step 1: collect children per parent via linked list, then sort
    # We need to build a sorted children structure.
    # First, count children per parent for allocation.
    num_children = np.zeros(n, dtype=np.int64)
    for i in range(n):
        if ancestor_ids[i] != i:
            num_children[ancestor_ids[i]] += 1

    # Build CSR offsets
    offsets = np.zeros(n + 1, dtype=np.int64)
    for i in range(n):
        offsets[i + 1] = offsets[i] + num_children[i]

    # Fill children by traversing linked list (first_child -> next_sibling)
    children = np.empty(offsets[n], dtype=np.int64)
    for i in range(n):
        pos = offsets[i]
        child = first_child_ids[i]
        if child == i:
            continue  # leaf node, no children
        while True:
            children[pos] = child
            pos += 1
            nxt = next_sibling_ids[child]
            if nxt == child:
                break
            child = nxt

    # Step 2: sort children of each parent by criterion using np.argsort
    for i in range(n):
        start = offsets[i]
        end = offsets[i + 1]
        count = end - start
        if count > 1:
            vals = np.empty(count, dtype=criterion_values.dtype)
            for j in range(count):
                vals[j] = criterion_values[children[start + j]]
            order = np.argsort(vals)
            if reverse:
                order = order[::-1].copy()
            sorted_children = np.empty(count, dtype=np.int64)
            for j in range(count):
                sorted_children[j] = children[start + order[j]]
            for j in range(count):
                children[start + j] = sorted_children[j]

    # Step 3: preorder DFS
    result = np.empty(n, dtype=np.int64)
    stack = np.empty(n, dtype=np.int64)

    # find roots and push in reverse order
    num_roots = 0
    for i in range(n):
        if ancestor_ids[i] == i:
            num_roots += 1

    root_ids = np.empty(num_roots, dtype=np.int64)
    ri = 0
    for i in range(n):
        if ancestor_ids[i] == i:
            root_ids[ri] = i
            ri += 1

    stack_top = 0
    for i in range(num_roots - 1, -1, -1):
        stack[stack_top] = root_ids[i]
        stack_top += 1

    result_idx = 0
    while stack_top > 0:
        stack_top -= 1
        node = stack[stack_top]
        result[result_idx] = node
        result_idx += 1

        # push children in reverse sorted order so first pops first
        start = offsets[node]
        end = offsets[node + 1]
        for j in range(end - 1, start - 1, -1):
            stack[stack_top] = children[j]
            stack_top += 1

    return result


def _alifestd_sort_children_asexual_slow_path(
    phylogeny_df: pd.DataFrame,
    criterion: str,
    reverse: bool = False,
) -> pd.DataFrame:
    """Implementation detail for `alifestd_sort_children_asexual`."""
    phylogeny_df.index = phylogeny_df["id"]

    # build children dict
    children = {}
    for idx in phylogeny_df.index:
        aid = phylogeny_df.at[idx, "ancestor_id"]
        if aid != idx:
            children.setdefault(aid, []).append(idx)

    # sort children by criterion column
    for parent_id in children:
        children[parent_id].sort(
            key=lambda c: phylogeny_df.at[c, criterion],
            reverse=reverse,
        )

    # preorder DFS
    roots = [
        idx
        for idx in phylogeny_df.index
        if phylogeny_df.at[idx, "ancestor_id"] == idx
    ]

    order = []
    stack = list(reversed(roots))
    while stack:
        node = stack.pop()
        order.append(node)
        for child in reversed(children.get(node, [])):
            stack.append(child)

    return phylogeny_df.loc[order].reset_index(drop=True)


def alifestd_sort_children_asexual(
    phylogeny_df: pd.DataFrame,
    criterion: str,
    reverse: bool = False,
    mutate: bool = False,
) -> pd.DataFrame:
    """Reorder rows so children are sorted by the given criterion column.

    Performs a preorder DFS traversal, visiting children of each node in
    order of ascending ``criterion`` column values. Set ``reverse=True``
    to sort descending (higher values first).

    The ``criterion`` column must already be present in the dataframe
    (e.g., added via ``alifestd_mark_num_leaves_asexual``).

    A topological sort will be applied if `phylogeny_df` is not topologically
    sorted. Dataframe reindexing (e.g., df.index) may be applied.

    Input dataframe is not mutated by this operation unless `mutate` set True.
    If mutate set True, operation does not occur in place; still use return
    value to get transformed phylogeny dataframe.

    Note: after sorting, ids will no longer be contiguous with respect to
    row indices. Call ``alifestd_assign_contiguous_ids`` on the result to
    reassign contiguous ids if needed.

    Parameters
    ----------
    phylogeny_df : pandas.DataFrame
        The phylogeny as a dataframe in alife standard format.
    criterion : str
        Name of the column to sort children by.
    reverse : bool, default False
        If True, sort descending (higher values first).
    mutate : bool, default False
        If True, allow mutation of the input dataframe.

    Returns
    -------
    pandas.DataFrame
        The phylogeny with rows reordered by sorted children traversal.

    See Also
    --------
    alifestd_sort_children_polars :
        Polars-based implementation.
    alifestd_ladderize_asexual :
        Convenience wrapper that sorts by ``num_leaves``.
    alifestd_assign_contiguous_ids :
        Reassign contiguous ids after reordering.
    """
    if not mutate:
        phylogeny_df = phylogeny_df.copy()

    phylogeny_df = alifestd_try_add_ancestor_id_col(phylogeny_df, mutate=True)

    if not alifestd_is_topologically_sorted(phylogeny_df):
        phylogeny_df = alifestd_topological_sort(phylogeny_df, mutate=True)

    if alifestd_has_contiguous_ids(phylogeny_df):
        ancestor_ids = phylogeny_df["ancestor_id"].to_numpy()
        criterion_values = (
            phylogeny_df[criterion]
            .to_numpy()
            .astype(
                np.float64,
            )
        )
        first_child_ids = _alifestd_mark_first_child_id_asexual_fast_path(
            ancestor_ids,
        )
        next_sibling_ids = _alifestd_mark_next_sibling_id_asexual_fast_path(
            ancestor_ids,
        )
        order = _alifestd_sort_children_asexual_fast_path(
            ancestor_ids,
            criterion_values,
            first_child_ids,
            next_sibling_ids,
            reverse=reverse,
        )
        return phylogeny_df.iloc[order].reset_index(drop=True)
    else:
        return _alifestd_sort_children_asexual_slow_path(
            phylogeny_df,
            criterion=criterion,
            reverse=reverse,
        )


_raw_description = f"""{os.path.basename(__file__)} | (phyloframe v{get_phyloframe_version()}/joinem v{joinem.__version__})

Reorder rows so children are sorted by the given ``--criterion`` column.

Performs a preorder DFS traversal, visiting children of each node in
order of ascending criterion values. Use ``--reverse`` to sort descending.

The criterion column must already be present in the input data.

Note: after sorting, ids will no longer be contiguous with respect to row
indices. Call ``alifestd_assign_contiguous_ids`` on the result to reassign
contiguous ids if needed.

Data is assumed to be in alife standard format.

Additional Notes
================
- Use `--eager-read` if modifying data file inplace.

- This CLI entrypoint is experimental and may be subject to change.

See Also
========
phyloframe.legacy._alifestd_sort_children_polars :
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
        dfcli_module="phyloframe.legacy._alifestd_sort_children_asexual",
        dfcli_version=get_phyloframe_version(),
    )
    parser.add_argument(
        "--criterion",
        required=False,
        default=None,
        type=str,
        help="column name to sort children by",
    )
    add_bool_arg(
        parser,
        "reverse",
        default=False,
        help="sort descending by criterion (default: False)",
    )
    return parser


if __name__ == "__main__":
    begin_prod_logging()

    parser = _create_parser()
    args, __ = parser.parse_known_args()
    with log_context_duration(
        "phyloframe.legacy._alifestd_sort_children_asexual", logging.info
    ):
        _run_dataframe_cli(
            base_parser=parser,
            output_dataframe_op=delegate_polars_implementation()(
                functools.partial(
                    alifestd_sort_children_asexual,
                    criterion=args.criterion,
                    reverse=args.reverse,
                ),
            ),
        )
