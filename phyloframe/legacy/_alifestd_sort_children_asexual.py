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
from ._alifestd_mark_num_children_asexual import (
    _alifestd_mark_num_children_asexual_fast_path,
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
    num_children: np.ndarray,
    reverse: bool = False,
) -> np.ndarray:
    """Implementation detail for `alifestd_sort_children_asexual`.

    Returns a permutation array giving the row order after sorting children
    by criterion values via preorder traversal.

    Uses first_child_id / next_sibling_id linked-list representation from
    PR #36 to traverse children of each node.

    Assumes contiguous ids and topological sorting.
    """
    n = len(ancestor_ids)
    if n == 0:
        return np.empty(0, dtype=np.int64)

    # Mutable copies of linked list for rewriting sorted order
    first_child = first_child_ids.copy()
    next_sib = next_sibling_ids.copy()

    # Step 1: sort children of each parent by rewriting linked list
    for parent in range(n):
        nc = num_children[parent]
        if nc <= 1:
            continue

        # Collect children via linked list
        buf = np.empty(nc, dtype=np.int64)
        child = first_child[parent]
        for j in range(nc):
            buf[j] = child
            child = next_sib[child]

        # Sort using np.argsort
        vals = np.empty(nc, dtype=criterion_values.dtype)
        for j in range(nc):
            vals[j] = criterion_values[buf[j]]
        order = np.argsort(vals)
        if reverse:
            order = order[::-1].copy()

        # Rewrite linked list to reflect sorted order
        first_child[parent] = buf[order[0]]
        for j in range(nc - 1):
            next_sib[buf[order[j]]] = buf[order[j + 1]]
        next_sib[buf[order[nc - 1]]] = buf[order[nc - 1]]

    # Step 2: compute subtree sizes (bottom-up)
    subtree_size = np.ones(n, dtype=np.int64)
    for i in range(n - 1, -1, -1):
        if ancestor_ids[i] != i:
            subtree_size[ancestor_ids[i]] += subtree_size[i]

    # Step 3: compute preorder positions (top-down via sorted linked list)
    position = np.zeros(n, dtype=np.int64)

    # Assign root positions
    offset = 0
    for i in range(n):
        if ancestor_ids[i] == i:
            position[i] = offset
            offset += subtree_size[i]

    # Assign child positions using sorted linked list
    for parent in range(n):
        child = first_child[parent]
        if child == parent:
            continue  # leaf
        offset = position[parent] + 1
        while child != parent:
            position[child] = offset
            offset += subtree_size[child]
            nxt = next_sib[child]
            child = nxt if nxt != child else parent

    return np.argsort(position)


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
        num_children = _alifestd_mark_num_children_asexual_fast_path(
            ancestor_ids,
        )
        order = _alifestd_sort_children_asexual_fast_path(
            ancestor_ids,
            criterion_values,
            first_child_ids,
            next_sibling_ids,
            num_children,
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
