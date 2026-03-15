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
from ._alifestd_has_contiguous_ids_polars import (
    alifestd_has_contiguous_ids_polars,
)
from ._alifestd_is_topologically_sorted_polars import (
    alifestd_is_topologically_sorted_polars,
)
from ._alifestd_mark_num_children_polars import (
    alifestd_mark_num_children_polars,
)
from ._alifestd_mark_roots_polars import alifestd_mark_roots_polars
from ._alifestd_try_add_ancestor_id_col_polars import (
    alifestd_try_add_ancestor_id_col_polars,
)


def alifestd_delete_unifurcating_roots_polars(
    phylogeny_df: pl.DataFrame,
) -> pl.DataFrame:
    """Pare record to bypass root nodes with only one descendant."""

    logging.info(
        "- alifestd_delete_unifurcating_roots_polars: adding ancestor_id col...",
    )
    phylogeny_df = alifestd_try_add_ancestor_id_col_polars(phylogeny_df)

    if phylogeny_df.lazy().limit(1).collect().is_empty():
        return phylogeny_df

    if not alifestd_has_contiguous_ids_polars(phylogeny_df):

        raise NotImplementedError(
            "non-contiguous ids not supported",
        )

    if not alifestd_is_topologically_sorted_polars(phylogeny_df):

        raise NotImplementedError(
            "non-topologically-sorted data not supported",
        )

    schema_names = phylogeny_df.lazy().collect_schema().names()
    if "num_children" not in schema_names:
        phylogeny_df = alifestd_mark_num_children_polars(phylogeny_df)
    if "is_root" not in schema_names:
        phylogeny_df = alifestd_mark_roots_polars(phylogeny_df)

    logging.info(
        "- alifestd_delete_unifurcating_roots_polars: identifying unifurcating roots...",
    )
    # Mark unifurcating roots
    phylogeny_df = phylogeny_df.with_columns(
        is_unifurcating_root=(
            (pl.col("num_children") == 1) & pl.col("is_root")
        ),
    )

    # For nodes whose ancestor is a unifurcating root, reparent to self
    # Use polars gather to look up ancestor's is_unifurcating_root flag
    phylogeny_df = phylogeny_df.with_columns(
        ancestor_id=pl.when(
            (pl.col("ancestor_id") != pl.col("id"))
            & pl.col("is_unifurcating_root").gather(pl.col("ancestor_id")),
        )
        .then(pl.col("id"))
        .otherwise(pl.col("ancestor_id")),
    )

    # Filter out unifurcating roots
    return phylogeny_df.filter(~pl.col("is_unifurcating_root"))


_raw_description = f"""{os.path.basename(__file__)} | (phyloframe v{get_phyloframe_version()}/joinem v{joinem.__version__})

Pare record to bypass root nodes with only one descendant.

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
        dfcli_module=(
            "phyloframe.legacy._alifestd_delete_unifurcating_roots_polars"
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
            "phyloframe.legacy._alifestd_delete_unifurcating_roots_polars",
            logging.info,
        ):
            _run_dataframe_cli(
                base_parser=parser,
                output_dataframe_op=(
                    alifestd_delete_unifurcating_roots_polars
                ),
            )
    except NotImplementedError as e:
        logging.error(
            "- polars op not yet implemented, use pandas op CLI instead",
        )
        raise e
