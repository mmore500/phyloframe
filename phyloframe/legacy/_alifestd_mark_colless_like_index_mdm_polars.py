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
from ._alifestd_mark_colless_like_index_mdm_asexual import (
    _colless_like_fast_path,
)
from ._alifestd_try_add_ancestor_id_col_polars import (
    alifestd_try_add_ancestor_id_col_polars,
)


def alifestd_mark_colless_like_index_mdm_polars(
    phylogeny_df: pl.DataFrame,
    mark_as: str = "colless_like_index_mdm",
) -> pl.DataFrame:
    """Add column `colless_like_index_mdm` with Colless-like index using
    mean deviation from the median (MDM) as dissimilarity.

    The output column name can be changed via the ``mark_as`` parameter.
    """

    logging.info(
        "- alifestd_mark_colless_like_index_mdm_polars: "
        "adding ancestor_id col...",
    )
    phylogeny_df = alifestd_try_add_ancestor_id_col_polars(phylogeny_df)

    if phylogeny_df.lazy().limit(1).collect().is_empty():
        return phylogeny_df.with_columns(
            pl.lit(0.0).cast(pl.Float64).alias(mark_as),
        )

    if not alifestd_has_contiguous_ids_polars(phylogeny_df):

        raise NotImplementedError(
            "non-contiguous ids not supported",
        )

    if not alifestd_is_topologically_sorted_polars(phylogeny_df):

        raise NotImplementedError(
            "non-topologically-sorted data not supported",
        )

    logging.info(
        "- alifestd_mark_colless_like_index_mdm_polars: "
        "extracting ancestor ids...",
    )
    ancestor_ids = (
        phylogeny_df.lazy()
        .select("ancestor_id")
        .collect()
        .to_series()
        .to_numpy()
    )

    logging.info(
        "- alifestd_mark_colless_like_index_mdm_polars: "
        "computing colless_like_index_mdm...",
    )
    return phylogeny_df.with_columns(
        pl.Series(
            _colless_like_fast_path(ancestor_ids, 0),
        ).alias(mark_as),
    )


_raw_description = f"""{os.path.basename(__file__)} | (phyloframe v{get_phyloframe_version()}/joinem v{joinem.__version__})

Add column `colless_like_index_mdm` with Colless-like index using mean deviation from the median (MDM) as dissimilarity.

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
            "phyloframe.legacy._alifestd_mark_colless_like_index_mdm_polars"
        ),
        dfcli_version=get_phyloframe_version(),
    )
    parser.add_argument(
        "--mark-as",
        default="colless_like_index_mdm",
        type=str,
        help="output column name (default: colless_like_index_mdm)",
    )
    return parser


if __name__ == "__main__":
    begin_prod_logging()

    parser = _create_parser()
    args, __ = parser.parse_known_args()

    try:
        with log_context_duration(
            "phyloframe.legacy"
            "._alifestd_mark_colless_like_index_mdm_polars",
            logging.info,
        ):
            _run_dataframe_cli(
                base_parser=parser,
                output_dataframe_op=functools.partial(
                    alifestd_mark_colless_like_index_mdm_polars,
                    mark_as=args.mark_as,
                ),
            )
    except NotImplementedError as e:
        logging.error(
            "- polars op not yet implemented, use pandas op CLI instead",
        )
        raise e
