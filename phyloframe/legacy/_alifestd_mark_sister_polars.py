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
from ._alifestd_assign_contiguous_ids_polars import (
    alifestd_assign_contiguous_ids_polars,
)
from ._alifestd_has_contiguous_ids_polars import (
    alifestd_has_contiguous_ids_polars,
)
from ._alifestd_is_topologically_sorted_polars import (
    alifestd_is_topologically_sorted_polars,
)
from ._alifestd_mark_is_left_child_polars import (
    alifestd_mark_is_left_child_polars,
)
from ._alifestd_mark_is_right_child_polars import (
    alifestd_mark_is_right_child_polars,
)
from ._alifestd_mark_left_child_polars import (
    alifestd_mark_left_child_polars,
)
from ._alifestd_mark_right_child_polars import (
    alifestd_mark_right_child_polars,
)
from ._alifestd_topological_sort_polars import (
    alifestd_topological_sort_polars,
)
from ._alifestd_try_add_ancestor_id_col_polars import (
    alifestd_try_add_ancestor_id_col_polars,
)


def alifestd_mark_sister_polars(
    phylogeny_df: pl.DataFrame,
) -> pl.DataFrame:
    """Add column `sister_id`, containing the id of each node's sibling.

    Root nodes will be marked with their own id.
    """

    logging.info(
        "- alifestd_mark_sister_polars: adding ancestor_id col...",
    )
    phylogeny_df = alifestd_try_add_ancestor_id_col_polars(phylogeny_df)

    if phylogeny_df.lazy().limit(1).collect().is_empty():
        return phylogeny_df.with_columns(
            sister_id=pl.lit(0).cast(pl.Int64),
        )

    logging.info(
        "- alifestd_mark_sister_polars: checking contiguous ids...",
    )
    if not alifestd_has_contiguous_ids_polars(phylogeny_df):
        phylogeny_df = alifestd_assign_contiguous_ids_polars(phylogeny_df)

    logging.info(
        "- alifestd_mark_sister_polars: checking topological sort...",
    )
    if not alifestd_is_topologically_sorted_polars(phylogeny_df):
        phylogeny_df = alifestd_topological_sort_polars(phylogeny_df)
        phylogeny_df = alifestd_assign_contiguous_ids_polars(phylogeny_df)

    schema_names = phylogeny_df.lazy().collect_schema().names()
    if "left_child_id" not in schema_names:
        phylogeny_df = alifestd_mark_left_child_polars(phylogeny_df)
    if "right_child_id" not in schema_names:
        phylogeny_df = alifestd_mark_right_child_polars(phylogeny_df)
    schema_names = phylogeny_df.lazy().collect_schema().names()
    if "is_left_child" not in schema_names:
        phylogeny_df = alifestd_mark_is_left_child_polars(phylogeny_df)
    if "is_right_child" not in schema_names:
        phylogeny_df = alifestd_mark_is_right_child_polars(phylogeny_df)

    logging.info(
        "- alifestd_mark_sister_polars: computing sister ids...",
    )
    return phylogeny_df.with_columns(
        sister_id=pl.when(pl.col("ancestor_id") == pl.col("id"))
        .then(pl.col("id"))
        .when(pl.col("is_left_child"))
        .then(pl.col("right_child_id").gather(pl.col("ancestor_id")))
        .otherwise(pl.col("left_child_id").gather(pl.col("ancestor_id"))),
    )


_raw_description = f"""{os.path.basename(__file__)} | (phyloframe v{get_phyloframe_version()}/joinem v{joinem.__version__})

Add column `sister_id`, containing the id of each node's sibling.

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
        dfcli_module="phyloframe.legacy._alifestd_mark_sister_polars",
        dfcli_version=get_phyloframe_version(),
    )
    return parser


if __name__ == "__main__":
    begin_prod_logging()

    parser = _create_parser()
    args, __ = parser.parse_known_args()

    try:
        with log_context_duration(
            "phyloframe.legacy._alifestd_mark_sister_polars",
            logging.info,
        ):
            _run_dataframe_cli(
                base_parser=parser,
                output_dataframe_op=alifestd_mark_sister_polars,
            )
    except NotImplementedError as e:
        logging.error(
            "- polars op not yet implemented, use pandas op CLI instead",
        )
        raise e
