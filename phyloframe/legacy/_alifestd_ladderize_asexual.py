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
from ._alifestd_mark_num_children_asexual import (
    _alifestd_mark_num_children_asexual_fast_path,
)
from ._alifestd_mark_num_leaves_asexual import (
    _alifestd_mark_num_leaves_asexual_fast_path,
    alifestd_mark_num_leaves_asexual,
)
from ._alifestd_topological_sort import alifestd_topological_sort
from ._alifestd_try_add_ancestor_id_col import (
    alifestd_try_add_ancestor_id_col,
)


@jit(nopython=True)
def _alifestd_ladderize_asexual_fast_path(
    ancestor_ids: np.ndarray,
    num_leaves: np.ndarray,
    num_children: np.ndarray,
    reverse: bool = False,
) -> np.ndarray:
    """Implementation detail for `alifestd_ladderize_asexual`.

    Returns a permutation array giving the ladderized row order.
    Assumes contiguous ids and topological sorting.
    """
    n = len(ancestor_ids)

    # Step 1: build CSR-style children array
    offsets = np.zeros(n + 1, dtype=np.int64)
    for i in range(n):
        offsets[i + 1] = offsets[i] + num_children[i]

    children = np.empty(offsets[n], dtype=np.int64)
    fill = offsets[:n].copy()
    for i in range(n):
        aid = ancestor_ids[i]
        if aid != i:
            children[fill[aid]] = i
            fill[aid] += 1

    # Step 2: sort children of each parent by num_leaves (insertion sort)
    for i in range(n):
        start = offsets[i]
        end = offsets[i + 1]
        for j in range(start + 1, end):
            key_child = children[j]
            key_val = num_leaves[key_child]
            k = j - 1
            if not reverse:
                while k >= start and num_leaves[children[k]] > key_val:
                    children[k + 1] = children[k]
                    k -= 1
            else:
                while k >= start and num_leaves[children[k]] < key_val:
                    children[k + 1] = children[k]
                    k -= 1
            children[k + 1] = key_child

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


def _alifestd_ladderize_asexual_slow_path(
    phylogeny_df: pd.DataFrame,
    reverse: bool = False,
) -> pd.DataFrame:
    """Implementation detail for `alifestd_ladderize_asexual`."""
    phylogeny_df = alifestd_mark_num_leaves_asexual(phylogeny_df, mutate=True)
    phylogeny_df.index = phylogeny_df["id"]

    # build children dict
    children = {}
    for idx in phylogeny_df.index:
        aid = phylogeny_df.at[idx, "ancestor_id"]
        if aid != idx:
            children.setdefault(aid, []).append(idx)

    # sort children by num_leaves
    for parent_id in children:
        children[parent_id].sort(
            key=lambda c: phylogeny_df.at[c, "num_leaves"],
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

    phylogeny_df.drop(columns=["num_leaves"], inplace=True)
    return phylogeny_df.loc[order].reset_index(drop=True)


def alifestd_ladderize_asexual(
    phylogeny_df: pd.DataFrame,
    reverse: bool = False,
    mutate: bool = False,
) -> pd.DataFrame:
    """Reorder rows so children are sorted by number of descendant leaves.

    By default, subtrees with fewer leaves come first (ascending). Set
    ``reverse=True`` to sort descending (more leaves first).

    A topological sort will be applied if `phylogeny_df` is not topologically
    sorted. Dataframe reindexing (e.g., df.index) may be applied.

    Input dataframe is not mutated by this operation unless `mutate` set True.
    If mutate set True, operation does not occur in place; still use return
    value to get transformed phylogeny dataframe.
    """
    if not mutate:
        phylogeny_df = phylogeny_df.copy()

    phylogeny_df = alifestd_try_add_ancestor_id_col(phylogeny_df, mutate=True)

    if not alifestd_is_topologically_sorted(phylogeny_df):
        phylogeny_df = alifestd_topological_sort(phylogeny_df, mutate=True)

    if alifestd_has_contiguous_ids(phylogeny_df):
        ancestor_ids = phylogeny_df["ancestor_id"].to_numpy()
        num_leaves = _alifestd_mark_num_leaves_asexual_fast_path(
            ancestor_ids,
        )
        num_children = _alifestd_mark_num_children_asexual_fast_path(
            ancestor_ids,
        )
        order = _alifestd_ladderize_asexual_fast_path(
            ancestor_ids,
            num_leaves,
            num_children,
            reverse=reverse,
        )
        return phylogeny_df.iloc[order].reset_index(drop=True)
    else:
        return _alifestd_ladderize_asexual_slow_path(
            phylogeny_df, reverse=reverse
        )


_raw_description = f"""{os.path.basename(__file__)} | (phyloframe v{get_phyloframe_version()}/joinem v{joinem.__version__})

Reorder rows so children are sorted by number of descendant leaves.

By default, subtrees with fewer leaves come first (ascending). Use
``--reverse`` to sort descending (more leaves first).

Data is assumed to be in alife standard format.

Additional Notes
================
- Use `--eager-read` if modifying data file inplace.

- This CLI entrypoint is experimental and may be subject to change.

See Also
========
phyloframe.legacy._alifestd_ladderize_polars :
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
        dfcli_module="phyloframe.legacy._alifestd_ladderize_asexual",
        dfcli_version=get_phyloframe_version(),
    )
    add_bool_arg(
        parser,
        "reverse",
        default=False,
        help="sort descending by leaf count (default: False)",
    )
    return parser


if __name__ == "__main__":
    begin_prod_logging()

    parser = _create_parser()
    args, __ = parser.parse_known_args()
    with log_context_duration(
        "phyloframe.legacy._alifestd_ladderize_asexual", logging.info
    ):
        _run_dataframe_cli(
            base_parser=parser,
            output_dataframe_op=delegate_polars_implementation()(
                functools.partial(
                    alifestd_ladderize_asexual,
                    reverse=args.reverse,
                ),
            ),
        )
