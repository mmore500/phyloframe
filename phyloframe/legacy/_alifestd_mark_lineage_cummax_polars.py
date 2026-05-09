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
from ._alifestd_is_topologically_sorted_polars import (
    alifestd_is_topologically_sorted_polars,
)
from ._alifestd_mark_lineage_cummax_asexual import (
    _alifestd_mark_lineage_cummax_asexual_fast_path,
)
from ._alifestd_try_add_ancestor_id_col_polars import (
    alifestd_try_add_ancestor_id_col_polars,
)


def alifestd_mark_lineage_cummax_polars(
    phylogeny_df: pl.DataFrame,
    values: typing.Union[str, pl.Expr],
    *,
    mark_as: str = "lineage_cummax",
    reverse: bool = False,
    skipna: bool = True,
) -> pl.DataFrame:
    """Add column with maximum of ``values`` along each lineage.

    With ``reverse=False`` (default), the result at each node is the
    maximum of ``values`` along the path from the root to that node,
    inclusive. With ``reverse=True``, the result at each node is the
    maximum of ``values`` over the entire clade rooted at that node,
    inclusive.

    See Also
    --------
    alifestd_mark_lineage_cummax_asexual :
        Pandas-based implementation.
    """
    if isinstance(values, str):
        schema_names = phylogeny_df.lazy().collect_schema().names()
        if values not in schema_names:
            raise ValueError(
                f"values column {values!r} not found in phylogeny_df",
            )
        values_expr = pl.col(values)
    else:
        values_expr = values

    phylogeny_df = alifestd_try_add_ancestor_id_col_polars(phylogeny_df)
    schema_names = phylogeny_df.lazy().collect_schema().names()
    if "ancestor_id" not in schema_names:
        raise NotImplementedError(
            "alifestd_mark_lineage_cummax_polars only supports asexual "
            "phylogenies.",
        )

    if phylogeny_df.lazy().limit(1).collect().is_empty():
        return phylogeny_df.with_columns(
            values_expr.alias(mark_as),
        )

    if not alifestd_has_contiguous_ids_polars(phylogeny_df):
        raise NotImplementedError(
            "non-contiguous ids not supported",
        )

    if not alifestd_is_topologically_sorted_polars(phylogeny_df):
        raise NotImplementedError(
            "non-topologically-sorted data not supported",
        )

    ancestor_ids = (
        phylogeny_df.lazy()
        .select("ancestor_id")
        .collect()
        .to_series()
        .to_numpy()
    )
    values_arr = (
        phylogeny_df.lazy()
        .select(values_expr)
        .collect()
        .to_series()
        .to_numpy()
    )
    if skipna:
        values_arr = np.nan_to_num(
            values_arr,
            nan=-np.inf,
            posinf=np.inf,
            neginf=-np.inf,
        )

    result = _alifestd_mark_lineage_cummax_asexual_fast_path(
        ancestor_ids,
        values_arr,
        reverse,
    )

    return phylogeny_df.with_columns(
        pl.Series(name=mark_as, values=result),
    )


_raw_description = f"""\
{os.path.basename(__file__)} | \
(phyloframe v{get_phyloframe_version()}/joinem v{joinem.__version__})

Add column with maximum of `values` along each lineage.

Data is assumed to be in alife standard format.

Additional Notes
================
- Use `--eager-read` if modifying data file inplace.

- This CLI entrypoint is experimental and may be subject to change.

See Also
========
phyloframe.legacy._alifestd_mark_lineage_cummax_asexual :
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
        dfcli_module="phyloframe.legacy._alifestd_mark_lineage_cummax_polars",
        dfcli_version=get_phyloframe_version(),
    )
    parser.add_argument(
        "--values",
        default=None,
        type=str,
        help="name of column to aggregate along lineage",
    )
    parser.add_argument(
        "--mark-as",
        default="lineage_cummax",
        type=str,
        help="output column name (default: lineage_cummax)",
    )
    add_bool_arg(
        parser,
        "reverse",
        default=False,
        help="aggregate over clade rooted at each node "
        "instead of along lineage from root",
    )
    add_bool_arg(
        parser,
        "skipna",
        default=True,
        help="treat NaN values as identity; "
        "use --no-skipna to propagate NaN instead",
    )
    return parser


if __name__ == "__main__":
    begin_prod_logging()

    parser = _create_parser()
    args, __ = parser.parse_known_args()

    try:
        with log_context_duration(
            "phyloframe.legacy._alifestd_mark_lineage_cummax_polars",
            logging.info,
        ):
            _run_dataframe_cli(
                base_parser=parser,
                output_dataframe_op=functools.partial(
                    alifestd_mark_lineage_cummax_polars,
                    values=args.values,
                    mark_as=args.mark_as,
                    reverse=args.reverse,
                    skipna=args.skipna,
                ),
            )
    except NotImplementedError as e:
        logging.error(
            "- polars op not yet implemented, use pandas op CLI instead",
        )
        raise e
