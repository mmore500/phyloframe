import argparse
import functools
import logging
import os
import typing

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
from ._alifestd_mark_lineage_delta_asexual import (
    _alifestd_mark_lineage_delta_asexual_fast_path,
)
from ._alifestd_try_add_ancestor_id_col_polars import (
    alifestd_try_add_ancestor_id_col_polars,
)


def alifestd_mark_lineage_delta_polars(
    phylogeny_df: pl.DataFrame,
    values: typing.Union[str, pl.Expr],
    *,
    basal_value: typing.Optional[typing.Any] = None,
    mark_as: str = "lineage_delta",
) -> pl.DataFrame:
    """Add column with the change in ``values`` from each node's parent.

    The result at each node is its value minus its parent's value. At a
    root, it is zero, or, if ``basal_value`` is given, the root's value
    minus ``basal_value``. Boolean values are cast to int8, producing
    deltas: +1 for a gain, -1 for a loss, and 0 for no change. See
    `alifestd_mark_lineage_delta_asexual`.

    Parameters
    ----------
    phylogeny_df : polars.DataFrame or polars.LazyFrame
        The phylogeny as a dataframe in alife standard format.

        Must represent an asexual phylogeny with contiguous ids.
    values : str or polars.Expr
        Column name or polars expression providing per-node values.
    basal_value : optional
        Value above each root, from which root deltas are measured; if not
        given, root deltas are zero.
    mark_as : str, default "lineage_delta"
        Output column name.

    See Also
    --------
    alifestd_mark_lineage_delta_asexual :
        Pandas-based implementation.
    alifestd_mark_lineage_cumsum_polars :
        Sums values along each lineage.
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
            "alifestd_mark_lineage_delta_polars only supports asexual "
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

    selected = (
        phylogeny_df.lazy()
        .select(pl.col("ancestor_id"), values_expr.alias("values"))
        .collect()
    )
    result = _alifestd_mark_lineage_delta_asexual_fast_path(
        selected["ancestor_id"].to_numpy(),
        selected["values"].to_numpy(),
        basal_value,
    )

    return phylogeny_df.with_columns(
        pl.Series(name=mark_as, values=result),
    )


_raw_description = f"""\
{os.path.basename(__file__)} | \
(phyloframe v{get_phyloframe_version()}/joinem v{joinem.__version__})

Add column with the change in `values` from each node's parent (at a root,
zero, or its value minus `--basal-value` if given).

Data is assumed to be in alife standard format.

Additional Notes
================
- Requires contiguous ids.

- Boolean values are taken as 0 and 1.

- Use `--eager-read` if modifying data file inplace.

- This CLI entrypoint is experimental and may be subject to change.

See Also
========
phyloframe.legacy._alifestd_mark_lineage_delta_asexual :
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
        dfcli_module="phyloframe.legacy._alifestd_mark_lineage_delta_polars",
        dfcli_version=get_phyloframe_version(),
    )
    parser.add_argument(
        "--values",
        default=None,
        type=str,
        help="name of column to take deltas of",
    )
    parser.add_argument(
        "--basal-value",
        default=None,
        type=float,
        help=(
            "value above each root, from which root deltas are measured "
            "(default: none, giving zero root deltas)"
        ),
    )
    parser.add_argument(
        "--mark-as",
        default="lineage_delta",
        type=str,
        help="output column name (default: lineage_delta)",
    )
    return parser


if __name__ == "__main__":
    begin_prod_logging()

    parser = _create_parser()
    args, __ = parser.parse_known_args()

    try:
        with log_context_duration(
            "phyloframe.legacy._alifestd_mark_lineage_delta_polars",
            logging.info,
        ):
            _run_dataframe_cli(
                base_parser=parser,
                output_dataframe_op=functools.partial(
                    alifestd_mark_lineage_delta_polars,
                    values=args.values,
                    basal_value=args.basal_value,
                    mark_as=args.mark_as,
                ),
            )
    except NotImplementedError as e:
        logging.error(
            "- polars op not yet implemented, use pandas op instead",
        )
        raise e
