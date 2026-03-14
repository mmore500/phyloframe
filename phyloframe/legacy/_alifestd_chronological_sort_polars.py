import argparse
import logging
import os

import joinem
from joinem._dataframe_cli import _add_parser_base, _run_dataframe_cli
import polars as pl

from .._auxlib._begin_prod_logging import begin_prod_logging
from .._auxlib._format_cli_description import format_cli_description
from .._auxlib._get_phyloframe_version import get_phyloframe_version
from .._auxlib._log_context_duration import log_context_duration


def alifestd_chronological_sort_polars(
    phylogeny_df: pl.DataFrame,
    how: str = "origin_time",
) -> pl.DataFrame:
    """Sort rows so all organisms appear in chronological order, default
    `origin_time`.
    """

    logging.info(
        "- alifestd_chronological_sort_polars: sorting by %s...",
        how,
    )
    return phylogeny_df.lazy().sort(how).collect()


_raw_description = f"""{os.path.basename(__file__)} | (phyloframe v{get_phyloframe_version()}/joinem v{joinem.__version__})

Sort rows so all organisms appear in chronological order.

Data is assumed to be in alife standard format.

Additional Notes
================
- Use `--eager-read` if modifying data file inplace.

- This CLI entrypoint is experimental and may be subject to change.
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
        dfcli_module=("phyloframe.legacy._alifestd_chronological_sort_polars"),
        dfcli_version=get_phyloframe_version(),
    )
    parser.add_argument(
        "--sort-by",
        type=str,
        default="origin_time",
        dest="sort_by",
        help="column name to sort by (default: origin_time)",
    )
    return parser


if __name__ == "__main__":
    begin_prod_logging()

    parser = _create_parser()
    args, __ = parser.parse_known_args()

    with log_context_duration(
        "phyloframe.legacy._alifestd_chronological_sort_polars",
        logging.info,
    ):
        import functools

        _run_dataframe_cli(
            base_parser=parser,
            output_dataframe_op=functools.partial(
                alifestd_chronological_sort_polars,
                how=args.sort_by,
            ),
        )
