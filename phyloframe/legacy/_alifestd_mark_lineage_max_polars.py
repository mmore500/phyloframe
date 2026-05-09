import argparse
import functools
import logging
import os
import typing

import joinem
from joinem._dataframe_cli import _add_parser_base, _run_dataframe_cli
import polars as pl

from .._auxlib._begin_prod_logging import begin_prod_logging
from .._auxlib._format_cli_description import format_cli_description
from .._auxlib._get_phyloframe_version import get_phyloframe_version
from .._auxlib._log_context_duration import log_context_duration
from ._alifestd_mark_lineage_cumsum_asexual import _OP_MAX
from ._alifestd_mark_lineage_cumsum_polars import (
    _alifestd_mark_lineage_op_polars,
)


def alifestd_mark_lineage_max_polars(
    phylogeny_df: pl.DataFrame,
    values: typing.Union[str, pl.Expr],
    *,
    mark_as: str = "lineage_max",
    reverse: bool = False,
    skipna: bool = True,
) -> pl.DataFrame:
    """Add column with maximum of ``values`` along each lineage.

    With ``reverse=False`` (default), the result at each node is the
    maximum of ``values`` along the path from the root to that node,
    inclusive. With ``reverse=True``, the result at each node is the
    maximum of ``values`` over the entire clade rooted at that node,
    inclusive.

    See Also
    --------
    alifestd_mark_lineage_max_asexual :
        Pandas-based implementation.
    """
    return _alifestd_mark_lineage_op_polars(
        phylogeny_df,
        values,
        _OP_MAX,
        mark_as=mark_as,
        reverse=reverse,
        skipna=skipna,
    )


_raw_description = f"""\
{os.path.basename(__file__)} | \
(phyloframe v{get_phyloframe_version()}/joinem v{joinem.__version__})

Add column with maximum of `values` along each lineage.

Data is assumed to be in alife standard format.

Additional Notes
================
- Use `--eager-read` if modifying data file inplace.

- This CLI entrypoint is experimental and may be subject to change.

See Also
========
phyloframe.legacy._alifestd_mark_lineage_max_asexual :
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
        dfcli_module="phyloframe.legacy._alifestd_mark_lineage_max_polars",
        dfcli_version=get_phyloframe_version(),
    )
    parser.add_argument(
        "--values",
        default=None,
        type=str,
        help="name of column to aggregate along lineage",
    )
    parser.add_argument(
        "--mark-as",
        default="lineage_max",
        type=str,
        help="output column name (default: lineage_max)",
    )
    parser.add_argument(
        "--reverse",
        action="store_true",
        help="aggregate over clade rooted at each node "
        "instead of along lineage from root",
    )
    parser.add_argument(
        "--no-skipna",
        action="store_true",
        help="propagate NaN values rather than treating as identity",
    )
    return parser


if __name__ == "__main__":
    begin_prod_logging()

    parser = _create_parser()
    args, __ = parser.parse_known_args()

    try:
        with log_context_duration(
            "phyloframe.legacy._alifestd_mark_lineage_max_polars",
            logging.info,
        ):
            _run_dataframe_cli(
                base_parser=parser,
                output_dataframe_op=functools.partial(
                    alifestd_mark_lineage_max_polars,
                    values=args.values,
                    mark_as=args.mark_as,
                    reverse=args.reverse,
                    skipna=not args.no_skipna,
                ),
            )
    except NotImplementedError as e:
        logging.error(
            "- polars op not yet implemented, use pandas op CLI instead",
        )
        raise e
