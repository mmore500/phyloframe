import argparse
import logging
import os
import warnings

import joinem
from joinem._dataframe_cli import _add_parser_base, _run_dataframe_cli
import polars as pl

from .._auxlib._begin_prod_logging import begin_prod_logging
from .._auxlib._format_cli_description import format_cli_description
from .._auxlib._get_phyloframe_version import get_phyloframe_version
from .._auxlib._log_context_duration import log_context_duration


def alifestd_unminify_polars(
    phylogeny_df: pl.DataFrame,
    *,
    dtype_id: pl.datatypes.DataType = pl.Int64,
) -> pl.DataFrame:
    """Restore alife standard format from a phylogeny minified by
    `alifestd_minify_polars`.

    Replaces the `ancestor_offset` column with an `ancestor_id` column and
    adds an `id` column corresponding to row position. Row order is
    preserved, so the returned phylogeny will have contiguous ids and be
    topologically sorted.

    No `ancestor_list` column is added. To add one, call
    `alifestd_try_add_ancestor_list_col_polars` on the result.

    Parameters
    ----------
    phylogeny_df : polars.DataFrame or polars.LazyFrame
        The phylogeny in minified format, with an `ancestor_offset` column.

        Any existing `id` or `ancestor_id` columns will be overwritten.
    dtype_id : pl.DataType, default pl.Int64
        Polars dtype for the `id` and `ancestor_id` columns.

    Returns
    -------
    polars.DataFrame or polars.LazyFrame
        The phylogeny in alife standard format.

    See Also
    --------
    alifestd_unminify_asexual :
        Pandas-based implementation.
    alifestd_minify_polars :
        Inverse operation, producing minified format.
    alifestd_try_add_ancestor_list_col_polars :
        Add alife standard `ancestor_list` column.
    """
    schema_names = phylogeny_df.lazy().collect_schema().names()
    for column in "id", "ancestor_id":
        if column in schema_names:
            warnings.warn(
                f"alifestd_unminify_polars overwrites existing {column}",
            )

    ids = pl.int_range(pl.len(), dtype=dtype_id)
    return phylogeny_df.select(
        ids.alias("id"),
        (ids - pl.col("ancestor_offset").cast(dtype_id)).alias("ancestor_id"),
        pl.exclude("id", "ancestor_id", "ancestor_offset"),
    )


_raw_description = f"""{os.path.basename(__file__)} | (phyloframe v{get_phyloframe_version()}/joinem v{joinem.__version__})

Restore alife standard format from a phylogeny minified by `alifestd_minify_polars`.

Replaces the `ancestor_offset` column with an `ancestor_id` column and adds an `id` column corresponding to row position.
Row order is preserved, so the output phylogeny will have contiguous ids and be topologically sorted.

No `ancestor_list` column is added.
To add one, chain with `alifestd_try_add_ancestor_list_col_polars` using the `phyloframe.legacy._alifestd_pipe_unary_ops_polars` entrypoint, e.g.,
    ls -1 "minified.pqt" \\
      | python3 -m phyloframe.legacy._alifestd_pipe_unary_ops_polars \\
        --op "pfl.alifestd_unminify_polars" \\
        --op "pfl.alifestd_try_add_ancestor_list_col_polars" \\
        "output.pqt"

Data is assumed to be in minified format.

Additional Notes
================
- Use `--eager-read` if modifying data file inplace.

- This CLI entrypoint is experimental and may be subject to change.

See Also
========
phyloframe.legacy._alifestd_unminify_asexual :
    CLI entrypoint for Pandas-based implementation.
phyloframe.legacy._alifestd_minify_polars :
    CLI entrypoint for inverse operation, producing minified format.
phyloframe.legacy._alifestd_try_add_ancestor_list_col_polars :
    CLI entrypoint to add alife standard `ancestor_list` column.
phyloframe.legacy._alifestd_pipe_unary_ops_polars :
    CLI entrypoint to chain multiple operations in a single process.
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
        dfcli_module="phyloframe.legacy._alifestd_unminify_polars",
        dfcli_version=get_phyloframe_version(),
    )
    return parser


if __name__ == "__main__":
    begin_prod_logging()

    parser = _create_parser()
    args, __ = parser.parse_known_args()

    try:
        with log_context_duration(
            "phyloframe.legacy._alifestd_unminify_polars",
            logging.info,
        ):
            _run_dataframe_cli(
                base_parser=parser,
                output_dataframe_op=alifestd_unminify_polars,
            )
    except NotImplementedError as e:
        logging.error(
            "- polars op not yet implemented, use pandas op CLI instead",
        )
        raise e
