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
from ._alifestd_mark_next_sibling_id_asexual import (
    _alifestd_mark_next_sibling_id_asexual_fast_path,
)


def alifestd_mark_next_sibling_id_polars(
    phylogeny_df: pl.DataFrame,
    mark_as: str = "next_sibling_id",
) -> pl.DataFrame:
    """Add column `next_sibling_id`, the next-highest id sharing the same
    parent.

    The output column name can be changed via the ``mark_as`` parameter.

    If no such sibling exists, marks own id.
    """

    if not alifestd_has_contiguous_ids_polars(phylogeny_df):

        raise NotImplementedError(
            "non-contiguous ids not supported",
        )

    if not alifestd_is_topologically_sorted_polars(phylogeny_df):

        raise NotImplementedError(
            "non-topologically-sorted data not supported",
        )

    logging.info(
        "- alifestd_mark_next_sibling_id_polars: extracting ancestor ids...",
    )
    ancestor_ids = (
        phylogeny_df.lazy()
        .select("ancestor_id")
        .collect()
        .to_series()
        .to_numpy()
    )

    logging.info(
        "- alifestd_mark_next_sibling_id_polars: "
        "tabulating next sibling ids...",
    )
    next_sibling_ids = _alifestd_mark_next_sibling_id_asexual_fast_path(
        ancestor_ids,
    )

    return phylogeny_df.with_columns(
        pl.Series(next_sibling_ids).alias(mark_as),
    )


_raw_description = f"""{os.path.basename(__file__)} | (phyloframe v{get_phyloframe_version()}/joinem v{joinem.__version__})

Add column `next_sibling_id`, the next-highest id sharing the same parent.

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
        dfcli_module="phyloframe.legacy._alifestd_mark_next_sibling_id_polars",
        dfcli_version=get_phyloframe_version(),
    )
    parser.add_argument(
        "--mark-as",
        default="next_sibling_id",
        type=str,
        help="output column name (default: next_sibling_id)",
    )
    return parser


if __name__ == "__main__":
    begin_prod_logging()

    parser = _create_parser()
    args, __ = parser.parse_known_args()

    try:
        with log_context_duration(
            "phyloframe.legacy._alifestd_mark_next_sibling_id_polars",
            logging.info,
        ):
            _run_dataframe_cli(
                base_parser=parser,
                output_dataframe_op=functools.partial(
                    alifestd_mark_next_sibling_id_polars, mark_as=args.mark_as
                ),
            )
    except NotImplementedError as e:
        logging.error(
            "- polars op not yet implemented, use pandas op CLI instead",
        )
        raise e
