import argparse
import functools
import logging
import os

import joinem
from joinem._dataframe_cli import _add_parser_base, _run_dataframe_cli
import pandas as pd

from .._auxlib._add_bool_arg import add_bool_arg
from .._auxlib._begin_prod_logging import begin_prod_logging
from .._auxlib._delegate_polars_implementation import (
    delegate_polars_implementation,
)
from .._auxlib._format_cli_description import format_cli_description
from .._auxlib._get_phyloframe_version import get_phyloframe_version
from .._auxlib._log_context_duration import log_context_duration
from ._alifestd_mark_num_leaves_asexual import (
    alifestd_mark_num_leaves_asexual,
)
from ._alifestd_sort_children_asexual import (
    alifestd_sort_children_asexual,
)


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

    Note: after ladderizing, ids will no longer be contiguous with respect to
    row indices. Call ``alifestd_assign_contiguous_ids`` on the result to
    reassign contiguous ids if needed.
    """
    if not mutate:
        phylogeny_df = phylogeny_df.copy()

    had_num_leaves = "num_leaves" in phylogeny_df.columns
    if not had_num_leaves:
        phylogeny_df = alifestd_mark_num_leaves_asexual(
            phylogeny_df,
            mutate=True,
        )
    phylogeny_df = alifestd_sort_children_asexual(
        phylogeny_df,
        criterion="num_leaves",
        reverse=reverse,
        mutate=True,
    )
    if not had_num_leaves:
        phylogeny_df.drop(columns=["num_leaves"], inplace=True)
    return phylogeny_df


_raw_description = f"""{os.path.basename(__file__)} | (phyloframe v{get_phyloframe_version()}/joinem v{joinem.__version__})

Reorder rows so children are sorted by number of descendant leaves.

By default, subtrees with fewer leaves come first (ascending). Use
``--reverse`` to sort descending (more leaves first).

Note: after ladderizing, ids will no longer be contiguous with respect to
row indices. Call ``alifestd_assign_contiguous_ids`` on the result to
reassign contiguous ids if needed.

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
