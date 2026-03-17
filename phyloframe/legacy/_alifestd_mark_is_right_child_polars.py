import argparse
import functools
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
from ._alifestd_mark_right_child_polars import (
    alifestd_mark_right_child_polars,
)
from ._alifestd_mark_roots_polars import alifestd_mark_roots_polars
from ._alifestd_try_add_ancestor_id_col_polars import (
    alifestd_try_add_ancestor_id_col_polars,
)


def alifestd_mark_is_right_child_polars(
    phylogeny_df: pl.DataFrame,
    *,
    mark_as: str = "is_right_child",
) -> pl.DataFrame:
    """Add column `is_right_child`, containing for each node whether it is the
    larger-id child.

    The output column name can be changed via the ``mark_as`` parameter.

    Root nodes will be marked False.
    """

    logging.info(
        "- alifestd_mark_is_right_child_polars: adding ancestor_id col...",
    )
    phylogeny_df = alifestd_try_add_ancestor_id_col_polars(phylogeny_df)

    if phylogeny_df.lazy().limit(1).collect().is_empty():
        return phylogeny_df.with_columns(
            pl.lit(False).alias(mark_as),
        )

    if not alifestd_has_contiguous_ids_polars(phylogeny_df):

        raise NotImplementedError(
            "non-contiguous ids not supported",
        )

    if not alifestd_is_topologically_sorted_polars(phylogeny_df):

        raise NotImplementedError(
            "non-topologically-sorted data not supported",
        )

    if "right_child_id" not in phylogeny_df.lazy().collect_schema().names():
        phylogeny_df = alifestd_mark_right_child_polars(phylogeny_df)

    if "is_root" not in phylogeny_df.lazy().collect_schema().names():
        phylogeny_df = alifestd_mark_roots_polars(phylogeny_df)

    logging.info(
        "- alifestd_mark_is_right_child_polars: "
        "computing is_right_child...",
    )
    return phylogeny_df.with_columns(
        (
            pl.col("right_child_id")
            .gather(pl.col("ancestor_id"))
            .eq(pl.col("id"))
            & ~pl.col("is_root")
        ).alias(mark_as),
    )


_raw_description = f"""{os.path.basename(__file__)} | (phyloframe v{get_phyloframe_version()}/joinem v{joinem.__version__})

Add column `is_right_child`, containing for each node whether it is the larger-id child.

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
            "phyloframe.legacy._alifestd_mark_is_right_child_polars"
        ),
        dfcli_version=get_phyloframe_version(),
    )
    parser.add_argument(
        "--mark-as",
        default="is_right_child",
        type=str,
        help="output column name (default: is_right_child)",
    )
    return parser


if __name__ == "__main__":
    begin_prod_logging()

    parser = _create_parser()
    args, __ = parser.parse_known_args()

    try:
        with log_context_duration(
            "phyloframe.legacy._alifestd_mark_is_right_child_polars",
            logging.info,
        ):
            _run_dataframe_cli(
                base_parser=parser,
                output_dataframe_op=functools.partial(
                    alifestd_mark_is_right_child_polars, mark_as=args.mark_as
                ),
            )
    except NotImplementedError as e:
        logging.error(
            "- polars op not yet implemented, use pandas op CLI instead",
        )
        raise e
