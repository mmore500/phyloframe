import argparse
import functools
import logging
import os
import typing

import joinem
from joinem._dataframe_cli import _add_parser_base, _run_dataframe_cli
import numpy as np
import polars as pl

from .._auxlib._add_bool_arg import add_bool_arg
from .._auxlib._begin_prod_logging import begin_prod_logging
from .._auxlib._format_cli_description import format_cli_description
from .._auxlib._get_phyloframe_version import get_phyloframe_version
from .._auxlib._log_context_duration import log_context_duration
from ._alifestd_has_contiguous_ids_polars import (
    alifestd_has_contiguous_ids_polars,
)
from ._alifestd_mark_children_min_asexual import (
    _alifestd_mark_children_min_asexual_fast_path,
)
from ._alifestd_try_add_ancestor_id_col_polars import (
    alifestd_try_add_ancestor_id_col_polars,
)


def alifestd_mark_children_min_polars(
    phylogeny_df: pl.DataFrame,
    values: typing.Union[str, pl.Expr],
    *,
    mark_as: str = "children_min",
    skipna: bool = True,
) -> pl.DataFrame:
    """Add column with minimum of ``values`` among each node's children.

    Nodes without children (i.e., leaves) are assigned null.

    Parameters
    ----------
    phylogeny_df : polars.DataFrame or polars.LazyFrame
        The phylogeny as a dataframe in alife standard format.

        Must represent an asexual phylogeny with contiguous ids.
    values : str or polars.Expr
        Column name or polars expression providing per-node values.
    mark_as : str, default "children_min"
        Output column name.
    skipna : bool, default True
        If True, NaN and null values are ignored; else propagate.

    Raises
    ------
    NotImplementedError
        If `phylogeny_df` is not asexual or has non-contiguous ids.

    See Also
    --------
    alifestd_mark_children_min_asexual :
        Pandas-based implementation.
    alifestd_mark_lineage_cummin_polars :
        Minimum along lineages, or over clades with ``reverse=True``.
    """
    if isinstance(values, str):
        schema_names = phylogeny_df.lazy().collect_schema().names()
        if values not in schema_names:
            raise ValueError(
                f"values column {values!r} not found in phylogeny_df",
            )
        values = pl.col(values)
    values_expr = values.cast(pl.Float64)

    phylogeny_df = alifestd_try_add_ancestor_id_col_polars(phylogeny_df)
    schema_names = phylogeny_df.lazy().collect_schema().names()
    if "ancestor_id" not in schema_names:
        raise NotImplementedError(
            "alifestd_mark_children_min_polars only supports asexual "
            "phylogenies.",
        )

    if phylogeny_df.lazy().limit(1).collect().is_empty():
        return phylogeny_df.with_columns(values_expr.alias(mark_as))

    if not alifestd_has_contiguous_ids_polars(phylogeny_df):
        raise NotImplementedError(
            "non-contiguous ids not supported",
        )

    logging.info(
        "- alifestd_mark_children_min_polars: extracting columns...",
    )
    arrays = (
        phylogeny_df.lazy()
        .select(pl.col("ancestor_id"), values_expr.alias("values"))
        .collect()
    )

    logging.info(
        "- alifestd_mark_children_min_polars: aggregating children...",
    )
    result = _alifestd_mark_children_min_asexual_fast_path(
        arrays["ancestor_id"].to_numpy(),
        arrays["values"].fill_null(np.nan).to_numpy(),
        skipna,
    )

    return phylogeny_df.with_columns(
        pl.Series(name=mark_as, values=result, nan_to_null=True),
    )


_raw_description = f"""\
{os.path.basename(__file__)} | \
(phyloframe v{get_phyloframe_version()}/joinem v{joinem.__version__})

Add column with minimum of `values` among each node's children
(null for leaves).

Data is assumed to be in alife standard format.

Additional Notes
================
- Use `--eager-read` if modifying data file inplace.

- This CLI entrypoint is experimental and may be subject to change.

See Also
========
phyloframe.legacy._alifestd_mark_children_min_asexual :
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
        dfcli_module="phyloframe.legacy._alifestd_mark_children_min_polars",
        dfcli_version=get_phyloframe_version(),
    )
    parser.add_argument(
        "--values",
        default=None,
        type=str,
        help="name of column to aggregate over children",
    )
    parser.add_argument(
        "--mark-as",
        default="children_min",
        type=str,
        help="output column name (default: children_min)",
    )
    add_bool_arg(
        parser,
        "skipna",
        default=True,
        help="ignore NaN values; use --no-skipna to propagate NaN instead",
    )
    return parser


if __name__ == "__main__":
    begin_prod_logging()

    parser = _create_parser()
    args, __ = parser.parse_known_args()

    try:
        with log_context_duration(
            "phyloframe.legacy._alifestd_mark_children_min_polars",
            logging.info,
        ):
            _run_dataframe_cli(
                base_parser=parser,
                output_dataframe_op=functools.partial(
                    alifestd_mark_children_min_polars,
                    values=args.values,
                    mark_as=args.mark_as,
                    skipna=args.skipna,
                ),
            )
    except NotImplementedError as e:
        logging.error(
            "- polars op not yet implemented, use pandas op CLI instead",
        )
        raise e
