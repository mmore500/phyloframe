import argparse
import logging
import os

import joinem
from joinem._dataframe_cli import _add_parser_base, _run_dataframe_cli
import polars as pl

from .._auxlib._add_bool_arg import add_bool_arg
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
from ._alifestd_make_ancestor_list_col_polars import (
    alifestd_make_ancestor_list_col_polars,
)
from ._alifestd_topological_sort_polars import (
    alifestd_topological_sort_polars,
)
from ._alifestd_try_add_ancestor_id_col_polars import (
    alifestd_try_add_ancestor_id_col_polars,
)


def alifestd_to_working_format_polars(
    phylogeny_df: pl.DataFrame,
    keep_ancestor_list: bool = False,
) -> pl.DataFrame:
    """Re-encode phylogeny_df to facilitate efficient analysis and
    transformation operations.

    The returned phylogeny dataframe will
    * be topologically sorted (i.e., organisms appear after all ancestors),
    * have contiguous ids (i.e., organisms' ids correspond to row number),
    * contain an integer datatype `ancestor_id` column if the phylogeny is
    asexual (i.e., a more performant representation of `ancestor_list`).

    Parameters
    ----------
    phylogeny_df : polars.DataFrame
        The phylogeny as a dataframe in alife standard format.
    keep_ancestor_list : bool, default False
        If True and `ancestor_list` was present in the input, regenerate the
        `ancestor_list` column from the (reassigned) `ancestor_id` column. The
        column is dropped during processing in all cases; it is only restored
        when this flag is set and the input already had it.

    See Also
    --------
    alifestd_to_working_format :
        Pandas-based implementation.
    """
    phylogeny_df = alifestd_try_add_ancestor_id_col_polars(phylogeny_df)

    schema_names = phylogeny_df.lazy().collect_schema().names()
    phylogeny_df = phylogeny_df.select(pl.exclude("ancestor_list"))

    if not alifestd_has_contiguous_ids_polars(phylogeny_df):
        phylogeny_df = phylogeny_df.pipe(alifestd_assign_contiguous_ids_polars)

    if not alifestd_is_topologically_sorted_polars(phylogeny_df):
        phylogeny_df = phylogeny_df.pipe(
            alifestd_topological_sort_polars,
        ).pipe(alifestd_assign_contiguous_ids_polars)

    if keep_ancestor_list and "ancestor_list" in schema_names:
        phylogeny_df = phylogeny_df.with_columns(
            ancestor_list=alifestd_make_ancestor_list_col_polars(
                phylogeny_df.lazy().select("id").collect().to_series(),
                phylogeny_df.lazy()
                .select("ancestor_id")
                .collect()
                .to_series(),
            ),
        )

    return phylogeny_df


_raw_description = f"""{os.path.basename(__file__)} | (phyloframe v{get_phyloframe_version()}/joinem v{joinem.__version__})

Re-encode phylogeny_df to facilitate efficient analysis and transformation operations.

The returned phylogeny dataframe will
- be topologically sorted (i.e., organisms appear after all ancestors),
- have contiguous ids (i.e., organisms' ids correspond to row number),
- contain an integer datatype `ancestor_id` column if the phylogeny is asexual (i.e., a more performant representation of `ancestor_list`).

Data is assumed to be in alife standard format.

Additional Notes
================
- Use `--eager-read` if modifying data file inplace.

- This CLI entrypoint is experimental and may be subject to change.

See Also
========
phyloframe.legacy._alifestd_to_working_format :
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
        dfcli_module="phyloframe.legacy._alifestd_to_working_format_polars",
        dfcli_version=get_phyloframe_version(),
    )
    add_bool_arg(
        parser,
        "keep-ancestor-list",
        default=False,
        help=(
            "regenerate the `ancestor_list` column from the reassigned "
            "`ancestor_id` column instead of dropping it (default: False)"
        ),
    )
    return parser


if __name__ == "__main__":
    begin_prod_logging()

    parser = _create_parser()
    args, __ = parser.parse_known_args()

    try:
        with log_context_duration(
            "phyloframe.legacy._alifestd_to_working_format_polars",
            logging.info,
        ):
            _run_dataframe_cli(
                base_parser=parser,
                output_dataframe_op=(
                    lambda df: alifestd_to_working_format_polars(
                        df,
                        keep_ancestor_list=args.keep_ancestor_list,
                    )
                ),
            )
    except NotImplementedError as e:
        logging.error(
            "- polars op not yet implemented, use pandas op CLI instead",
        )
        raise e
