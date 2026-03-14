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
from .._auxlib._log_context_duration import log_context_duration
from ._alifestd_has_contiguous_ids import alifestd_has_contiguous_ids
from ._alifestd_is_topologically_sorted import (
    alifestd_is_topologically_sorted,
)
from ._alifestd_mark_node_depth_asexual import (
    _alifestd_calc_node_depth_asexual_contiguous,
)
from ._alifestd_topological_sort import alifestd_topological_sort
from ._alifestd_try_add_ancestor_id_col import (
    alifestd_try_add_ancestor_id_col,
)


def _alifestd_sort_children_asexual_fast_path(
    ancestor_ids: np.ndarray,
    criterion_values: np.ndarray,
    node_depths: np.ndarray,
    reverse: bool = False,
) -> np.ndarray:
    """Implementation detail for `alifestd_sort_children_asexual`.

    Returns a permutation array giving the row order after sorting children
    by criterion values.

    Lexsorts by (depth, ancestor_id, criterion) to maintain topological
    order while sorting siblings by criterion.

    Assumes contiguous ids and topological sorting.
    """
    if reverse:
        criterion_values = -criterion_values
    return np.lexsort((criterion_values, ancestor_ids, node_depths))


def _alifestd_sort_children_asexual_slow_path(
    phylogeny_df: pd.DataFrame,
    criterion: str,
    reverse: bool = False,
) -> pd.DataFrame:
    """Implementation detail for `alifestd_sort_children_asexual`."""
    phylogeny_df.index = phylogeny_df["id"]

    depth = {}
    for idx in phylogeny_df.index:
        aid = phylogeny_df.at[idx, "ancestor_id"]
        depth[idx] = 0 if aid == idx else depth[aid] + 1

    sign = -1 if reverse else 1
    order = sorted(
        phylogeny_df.index,
        key=lambda idx: (
            depth[idx],
            phylogeny_df.at[idx, "ancestor_id"],
            sign * phylogeny_df.at[idx, criterion],
        ),
    )

    return phylogeny_df.loc[order].reset_index(drop=True)


def alifestd_sort_children_asexual(
    phylogeny_df: pd.DataFrame,
    criterion: str,
    reverse: bool = False,
    mutate: bool = False,
) -> pd.DataFrame:
    """Reorder rows so children are sorted by the given criterion column,
    gathering children into contiguous rows.

    Reorders rows so that among siblings, they appear in order of
    ascending ``criterion`` column values. Set ``reverse=True`` to sort
    descending (higher values first).

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
        node_depths = _alifestd_calc_node_depth_asexual_contiguous(
            ancestor_ids,
        )
        order = _alifestd_sort_children_asexual_fast_path(
            ancestor_ids,
            criterion_values,
            node_depths,
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

Reorder rows so children are sorted by the given ``--criterion`` column,
gathering children into contiguous rows.

Reorders rows so that among siblings, they appear in order of ascending
criterion values. Use ``--reverse`` to sort descending.

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
