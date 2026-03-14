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
from ._alifestd_mark_oldest_root_polars import (
    alifestd_mark_oldest_root_polars,
)
from ._alifestd_mark_roots_polars import alifestd_mark_roots_polars
from ._alifestd_try_add_ancestor_id_col_polars import (
    alifestd_try_add_ancestor_id_col_polars,
)


def alifestd_join_roots_polars(
    phylogeny_df: pl.DataFrame,
) -> pl.DataFrame:
    """Point all other roots to oldest root, measured by lowest `origin_time`
    (if available) or otherwise lowest `id`.
    """

    logging.info(
        "- alifestd_join_roots_polars: adding ancestor_id col...",
    )
    phylogeny_df = alifestd_try_add_ancestor_id_col_polars(phylogeny_df)

    if phylogeny_df.lazy().select(pl.len()).collect().item() <= 1:
        return phylogeny_df

    logging.info(
        "- alifestd_join_roots_polars: marking roots...",
    )
    if "is_root" not in phylogeny_df.lazy().collect_schema().names():
        phylogeny_df = alifestd_mark_roots_polars(phylogeny_df)

    logging.info(
        "- alifestd_join_roots_polars: marking oldest root...",
    )
    if "is_oldest_root" not in phylogeny_df.lazy().collect_schema().names():
        phylogeny_df = alifestd_mark_oldest_root_polars(phylogeny_df)

    df = phylogeny_df.lazy().collect()
    global_root_id = (
        df.lazy()
        .filter(pl.col("is_oldest_root"))
        .select("id")
        .collect()
        .item()
    )

    logging.info(
        "- alifestd_join_roots_polars: joining roots...",
    )
    df = df.with_columns(
        ancestor_id=pl.when(pl.col("is_root"))
        .then(pl.lit(global_root_id))
        .otherwise(pl.col("ancestor_id")),
        is_root=pl.col("is_oldest_root"),
    )

    return df


_raw_description = f"""{os.path.basename(__file__)} | (phyloframe v{get_phyloframe_version()}/joinem v{joinem.__version__})

Point all other roots to oldest root, measured by lowest `origin_time` (if available) or otherwise lowest `id`.

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
        dfcli_module="phyloframe.legacy._alifestd_join_roots_polars",
        dfcli_version=get_phyloframe_version(),
    )
    return parser


if __name__ == "__main__":
    begin_prod_logging()

    parser = _create_parser()
    args, __ = parser.parse_known_args()

    try:
        with log_context_duration(
            "phyloframe.legacy._alifestd_join_roots_polars",
            logging.info,
        ):
            _run_dataframe_cli(
                base_parser=parser,
                output_dataframe_op=alifestd_join_roots_polars,
            )
    except NotImplementedError as e:
        logging.error(
            "- polars op not yet implemented, use pandas op CLI instead",
        )
        raise e
