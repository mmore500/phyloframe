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
from ._alifestd_mark_num_leaves_polars import (
    alifestd_mark_num_leaves_polars,
)
from ._alifestd_sort_children_polars import (
    alifestd_sort_children_polars,
)


def alifestd_ladderize_polars(
    phylogeny_df: pl.DataFrame,
    reverse: bool = False,
) -> pl.DataFrame:
    """Reorder rows so children are sorted by number of descendant leaves,
    gathering children into contiguous rows.

    By default, subtrees with fewer leaves come first (ascending). Set
    ``reverse=True`` to sort descending (more leaves first).

    Note: after ladderizing, ids will no longer be contiguous with respect to
    row indices. Call ``alifestd_assign_contiguous_ids_polars`` on the result
    to reassign contiguous ids if needed.

    Parameters
    ----------
    phylogeny_df : polars.DataFrame
        The phylogeny as a dataframe in alife standard format.

        Must represent an asexual phylogeny with contiguous ids and
        topologically sorted rows.

    reverse : bool, default False
        If True, sort descending (more leaves first).

    Returns
    -------
    polars.DataFrame
        The phylogeny with rows reordered in ladderized order.

    Raises
    ------
    NotImplementedError
        If ids are not contiguous or rows are not topologically sorted.

    See Also
    --------
    alifestd_ladderize_asexual :
        Pandas-based implementation.
    """

    if "num_leaves" not in phylogeny_df.collect_schema().names():
        phylogeny_df = alifestd_mark_num_leaves_polars(phylogeny_df)
    return alifestd_sort_children_polars(
        phylogeny_df,
        criterion="num_leaves",
        reverse=reverse,
    )


_raw_description = f"""{os.path.basename(__file__)} | (phyloframe v{get_phyloframe_version()}/joinem v{joinem.__version__})

Reorder rows so children are sorted by number of descendant leaves,
gathering children into contiguous rows.

By default, subtrees with fewer leaves come first (ascending). Use
``--reverse`` to sort descending (more leaves first).

Note: after ladderizing, ids will no longer be contiguous with respect to
row indices. Call ``alifestd_assign_contiguous_ids_polars`` on the result to
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
phyloframe.legacy._alifestd_ladderize_asexual :
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
        dfcli_module="phyloframe.legacy._alifestd_ladderize_polars",
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

    try:
        with log_context_duration(
            "phyloframe.legacy._alifestd_ladderize_polars",
            logging.info,
        ):
            _run_dataframe_cli(
                base_parser=parser,
                output_dataframe_op=functools.partial(
                    alifestd_ladderize_polars,
                    reverse=args.reverse,
                ),
            )
    except NotImplementedError as e:
        logging.error(
            "- polars op not yet implemented, use pandas op CLI instead",
        )
        raise e
