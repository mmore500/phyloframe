import argparse
import functools
import logging
import os

import joinem
from joinem._dataframe_cli import _add_parser_base, _run_dataframe_cli
import polars as pl

from .._auxlib._add_bool_arg import add_bool_arg
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
from ._alifestd_mark_first_child_id_asexual import (
    _alifestd_mark_first_child_id_asexual_fast_path,
)
from ._alifestd_mark_next_sibling_id_asexual import (
    _alifestd_mark_next_sibling_id_asexual_fast_path,
)
from ._alifestd_mark_num_children_asexual import (
    _alifestd_mark_num_children_asexual_fast_path,
)
from ._alifestd_sort_children_asexual import (
    _alifestd_sort_children_asexual_fast_path,
)


def alifestd_sort_children_polars(
    phylogeny_df: pl.DataFrame,
    criterion: str,
    reverse: bool = False,
) -> pl.DataFrame:
    """Reorder rows so children are sorted by the given criterion column.

    Performs a preorder DFS traversal, visiting children of each node in
    order of ascending ``criterion`` column values. Set ``reverse=True``
    to sort descending (higher values first).

    The ``criterion`` column must already be present in the dataframe.

    Note: after sorting, ids will no longer be contiguous with respect to
    row indices. Call ``alifestd_assign_contiguous_ids_polars`` on the
    result to reassign contiguous ids if needed.

    Parameters
    ----------
    phylogeny_df : polars.DataFrame
        The phylogeny as a dataframe in alife standard format.

        Must represent an asexual phylogeny with contiguous ids and
        topologically sorted rows.

    criterion : str
        Name of the column to sort children by.

    reverse : bool, default False
        If True, sort descending (higher values first).

    Returns
    -------
    polars.DataFrame
        The phylogeny with rows reordered by sorted children traversal.

    Raises
    ------
    NotImplementedError
        If ids are not contiguous or rows are not topologically sorted.

    See Also
    --------
    alifestd_sort_children_asexual :
        Pandas-based implementation.
    alifestd_ladderize_polars :
        Convenience wrapper that sorts by ``num_leaves``.
    alifestd_assign_contiguous_ids_polars :
        Reassign contiguous ids after reordering.
    """

    logging.info(
        "- alifestd_sort_children_polars: checking contiguous ids...",
    )
    if not alifestd_has_contiguous_ids_polars(phylogeny_df):
        raise NotImplementedError("non-contiguous ids not yet supported")

    logging.info(
        "- alifestd_sort_children_polars: checking topological sort...",
    )
    if not alifestd_is_topologically_sorted_polars(phylogeny_df):
        raise NotImplementedError(
            "topologically unsorted rows not yet supported",
        )

    if phylogeny_df.lazy().limit(1).collect().is_empty():
        return phylogeny_df

    logging.info(
        "- alifestd_sort_children_polars: extracting data...",
    )
    ancestor_ids = (
        phylogeny_df.lazy()
        .select("ancestor_id")
        .collect()
        .to_series()
        .to_numpy()
    )
    criterion_values = (
        phylogeny_df.lazy()
        .select(criterion)
        .collect()
        .to_series()
        .to_numpy()
        .astype(float)
    )

    logging.info(
        "- alifestd_sort_children_polars: computing child structure...",
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

    logging.info(
        "- alifestd_sort_children_polars: computing sorted order...",
    )
    order = _alifestd_sort_children_asexual_fast_path(
        ancestor_ids,
        criterion_values,
        first_child_ids,
        next_sibling_ids,
        num_children,
        reverse=reverse,
    )

    return phylogeny_df.lazy().collect()[order.tolist()]


_raw_description = f"""{os.path.basename(__file__)} | (phyloframe v{get_phyloframe_version()}/joinem v{joinem.__version__})

Reorder rows so children are sorted by the given ``--criterion`` column.

Performs a preorder DFS traversal, visiting children of each node in
order of ascending criterion values. Use ``--reverse`` to sort descending.

The criterion column must already be present in the input data.

Note: after sorting, ids will no longer be contiguous with respect to row
indices. Call ``alifestd_assign_contiguous_ids_polars`` on the result to
reassign contiguous ids if needed.

Data is assumed to be in alife standard format.

Additional Notes
================
- Requires 'ancestor_id' column to be present in input DataFrame.
Otherwise, no action is taken.

- Use `--eager-read` if modifying data file inplace.

- This CLI entrypoint is experimental and may be subject to change.

See Also
========
phyloframe.legacy._alifestd_sort_children_asexual :
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
        dfcli_module="phyloframe.legacy._alifestd_sort_children_polars",
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

    try:
        with log_context_duration(
            "phyloframe.legacy._alifestd_sort_children_polars",
            logging.info,
        ):
            _run_dataframe_cli(
                base_parser=parser,
                output_dataframe_op=functools.partial(
                    alifestd_sort_children_polars,
                    criterion=args.criterion,
                    reverse=args.reverse,
                ),
            )
    except NotImplementedError as e:
        logging.error(
            "- polars op not yet implemented, use pandas op CLI instead",
        )
        raise e
