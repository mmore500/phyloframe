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
from ._alifestd_mark_ancestor_origin_time_polars import (
    alifestd_mark_ancestor_origin_time_polars,
)


def alifestd_mark_origin_time_delta_polars(
    phylogeny_df: pl.DataFrame,
) -> pl.DataFrame:
    """Add columns `origin_time_delta` and `ancestor_origin_time`.

    Dataframe must provide column `origin_time`.
    """

    logging.info(
        "- alifestd_mark_origin_time_delta_polars: marking ancestor origin times...",
    )
    phylogeny_df = alifestd_mark_ancestor_origin_time_polars(phylogeny_df)

    logging.info(
        "- alifestd_mark_origin_time_delta_polars: calculating time deltas...",
    )
    return phylogeny_df.with_columns(
        origin_time_delta=(
            pl.col("origin_time") - pl.col("ancestor_origin_time")
        ),
    )


_raw_description = f"""\
{os.path.basename(__file__)} | \
(phyloframe v{get_phyloframe_version()}/joinem v{joinem.__version__})

Add columns `origin_time_delta` and `ancestor_origin_time`.

Data is assumed to be in alife standard format.

Additional Notes
================
- Use `--eager-read` if modifying data file inplace.

- This CLI entrypoint is experimental and may be subject to change.

See Also
========
phyloframe.legacy._alifestd_mark_origin_time_delta_asexual :
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
        dfcli_module=(
            "phyloframe.legacy._alifestd_mark_origin_time_delta_polars"
        ),
        dfcli_version=get_phyloframe_version(),
    )
    return parser


if __name__ == "__main__":
    begin_prod_logging()

    parser = _create_parser()
    args, __ = parser.parse_known_args()

    try:
        with log_context_duration(
            "phyloframe.legacy._alifestd_mark_origin_time_delta_polars",
            logging.info,
        ):
            _run_dataframe_cli(
                base_parser=parser,
                output_dataframe_op=(alifestd_mark_origin_time_delta_polars),
            )
    except NotImplementedError as e:
        logging.error(
            "- polars op not yet implemented, use pandas op CLI instead",
        )
        raise e
