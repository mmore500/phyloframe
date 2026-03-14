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
from ._alifestd_mark_roots_polars import alifestd_mark_roots_polars


def alifestd_mark_oldest_root_polars(
    phylogeny_df: pl.DataFrame,
) -> pl.DataFrame:
    """Point all other roots to oldest root, measured by lowest
    `origin_time` (if available) or otherwise lowest `id`.
    """

    schema_names = phylogeny_df.lazy().collect_schema().names()

    if "is_root" not in schema_names:
        logging.info(
            "- alifestd_mark_oldest_root_polars: marking roots...",
        )
        phylogeny_df = alifestd_mark_roots_polars(phylogeny_df)

    n = phylogeny_df.lazy().select(pl.len()).collect().item()
    if n <= 1:
        return phylogeny_df.with_columns(
            is_oldest_root=pl.lit(True),
        )

    logging.info(
        "- alifestd_mark_oldest_root_polars: " "finding oldest root...",
    )
    roots = phylogeny_df.lazy().filter(pl.col("is_root")).collect()

    schema_names = phylogeny_df.lazy().collect_schema().names()
    if "origin_time" in schema_names:
        oldest_root_id = (
            roots.sort(["origin_time", "id"]).select("id").item(0, 0)
        )
    else:
        oldest_root_id = roots.select(pl.col("id").min()).item()

    return phylogeny_df.with_columns(
        is_oldest_root=(pl.col("id") == oldest_root_id),
    )


_raw_description = f"""\
{os.path.basename(__file__)} | \
(phyloframe v{get_phyloframe_version()}/joinem v{joinem.__version__})

Mark the oldest root row with an `is_oldest_root` column.

Data is assumed to be in alife standard format.

Additional Notes
================
- Use `--eager-read` if modifying data file inplace.

- This CLI entrypoint is experimental and may be subject to change.

See Also
========
phyloframe.legacy._alifestd_mark_oldest_root :
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
        dfcli_module=("phyloframe.legacy._alifestd_mark_oldest_root_polars"),
        dfcli_version=get_phyloframe_version(),
    )
    return parser


if __name__ == "__main__":
    begin_prod_logging()

    parser = _create_parser()
    args, __ = parser.parse_known_args()

    try:
        with log_context_duration(
            "phyloframe.legacy._alifestd_mark_oldest_root_polars",
            logging.info,
        ):
            _run_dataframe_cli(
                base_parser=parser,
                output_dataframe_op=alifestd_mark_oldest_root_polars,
            )
    except NotImplementedError as e:
        logging.error(
            "- polars op not yet implemented, use pandas op CLI instead",
        )
        raise e
