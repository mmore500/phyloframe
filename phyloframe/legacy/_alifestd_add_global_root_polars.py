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
from ._alifestd_try_add_ancestor_id_col_polars import (
    alifestd_try_add_ancestor_id_col_polars,
)


def alifestd_add_global_root_polars(
    phylogeny_df: pl.DataFrame,
) -> pl.DataFrame:
    """Add a new global root node that all existing roots point to."""

    logging.info(
        "- alifestd_add_global_root_polars: adding ancestor_id col...",
    )
    phylogeny_df = alifestd_try_add_ancestor_id_col_polars(phylogeny_df)

    schema_names = phylogeny_df.lazy().collect_schema().names()

    if phylogeny_df.lazy().limit(1).collect().is_empty():
        new_root_id = 0
    else:
        new_root_id = (
            phylogeny_df.lazy().select(pl.col("id").max()).collect().item() + 1
        )

    logging.info(
        "- alifestd_add_global_root_polars: reparenting roots...",
    )
    # Point existing roots to new root
    phylogeny_df = phylogeny_df.with_columns(
        ancestor_id=pl.when(pl.col("id") == pl.col("ancestor_id"))
        .then(pl.lit(new_root_id))
        .otherwise(pl.col("ancestor_id")),
    )

    # Build new root row
    new_root_data = {"id": [new_root_id], "ancestor_id": [new_root_id]}
    for col_name in schema_names:
        if col_name not in new_root_data:
            new_root_data[col_name] = [None]

    new_root_df = pl.DataFrame(
        new_root_data,
        schema=phylogeny_df.schema,
    )

    return pl.concat([phylogeny_df, new_root_df], how="align")


_raw_description = f"""{os.path.basename(__file__)} | (phyloframe v{get_phyloframe_version()}/joinem v{joinem.__version__})

Add a new global root node that all existing roots point to.

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
        dfcli_module=("phyloframe.legacy._alifestd_add_global_root_polars"),
        dfcli_version=get_phyloframe_version(),
    )
    return parser


if __name__ == "__main__":
    begin_prod_logging()

    parser = _create_parser()
    args, __ = parser.parse_known_args()

    try:
        with log_context_duration(
            "phyloframe.legacy._alifestd_add_global_root_polars",
            logging.info,
        ):
            _run_dataframe_cli(
                base_parser=parser,
                output_dataframe_op=alifestd_add_global_root_polars,
            )
    except NotImplementedError as e:
        logging.error(
            "- polars op not yet implemented, use pandas op CLI instead",
        )
        raise e
