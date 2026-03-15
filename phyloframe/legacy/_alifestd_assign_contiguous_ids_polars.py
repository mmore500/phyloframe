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
from .._auxlib._preserve_id_dtypes_polars import (
    preserve_id_dtypes_polars,
)
from ._alifestd_assign_contiguous_ids import _reassign_ids_asexual
from ._alifestd_try_add_ancestor_id_col_polars import (
    alifestd_try_add_ancestor_id_col_polars,
)


@preserve_id_dtypes_polars
def alifestd_assign_contiguous_ids_polars(
    phylogeny_df: pl.DataFrame,
) -> pl.DataFrame:
    """Reassign so each organism's id corresponds to its row number.

    Organisms retain the same row location; only id numbers change."""
    schema_names = phylogeny_df.lazy().collect_schema().names()

    if "ancestor_list" in schema_names:
        raise NotImplementedError(
            "ancestor_list column not supported, drop it first",
        )

    if "ancestor_id" not in schema_names:
        phylogeny_df = alifestd_try_add_ancestor_id_col_polars(phylogeny_df)

    id_arr = phylogeny_df.lazy().select("id").collect().to_series().to_numpy()
    ancestor_id_arr = (
        phylogeny_df.lazy()
        .select("ancestor_id")
        .collect()
        .to_series()
        .to_numpy()
    )
    new_ancestor_ids = _reassign_ids_asexual(id_arr, ancestor_id_arr)

    return (
        phylogeny_df.drop("id")
        .with_row_index("id")
        .with_columns(
            ancestor_id=pl.Series(new_ancestor_ids),
        )
    )


_raw_description = f"""{os.path.basename(__file__)} | (phyloframe v{get_phyloframe_version()}/joinem v{joinem.__version__})

Reassign so each organism's id corresponds to its row number.

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
        dfcli_module="phyloframe.legacy._alifestd_assign_contiguous_ids_polars",
        dfcli_version=get_phyloframe_version(),
    )
    return parser


if __name__ == "__main__":
    begin_prod_logging()

    parser = _create_parser()
    args, __ = parser.parse_known_args()

    try:
        with log_context_duration(
            "phyloframe.legacy._alifestd_assign_contiguous_ids_polars",
            logging.info,
        ):
            _run_dataframe_cli(
                base_parser=parser,
                output_dataframe_op=alifestd_assign_contiguous_ids_polars,
            )
    except NotImplementedError as e:
        logging.error(
            "- polars op not yet implemented, use pandas op CLI instead",
        )
        raise e
