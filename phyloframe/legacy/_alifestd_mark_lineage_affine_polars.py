import argparse
import functools
import logging
import os
import typing

import joinem
from joinem._dataframe_cli import _add_parser_base, _run_dataframe_cli
import numpy as np
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
from ._alifestd_mark_lineage_affine_asexual import (
    _alifestd_mark_lineage_affine_asexual_fast_path,
)
from ._alifestd_try_add_ancestor_id_col_polars import (
    alifestd_try_add_ancestor_id_col_polars,
)


def alifestd_mark_lineage_affine_polars(
    phylogeny_df: pl.DataFrame,
    scale: typing.Union[str, pl.Expr],
    offset: typing.Union[str, pl.Expr],
    *,
    mark_as: str = "lineage_affine",
) -> pl.DataFrame:
    """Add column evaluating an affine recurrence down each lineage.

    The result at each node is ``scale * result[ancestor] + offset``,
    and ``offset`` at roots. For example, a unit ``scale`` gives the
    cumulative sum of ``offset`` along each lineage, and a zero ``offset``
    (except one at roots) gives the cumulative product of ``scale``.

    Parameters
    ----------
    phylogeny_df : polars.DataFrame or polars.LazyFrame
        The phylogeny as a dataframe in alife standard format.

        Must represent an asexual phylogeny with contiguous,
        topologically sorted ids.
    scale : str or polars.Expr
        Column name or expression multiplying the ancestor's result.
    offset : str or polars.Expr
        Column name or expression added to the multiplied ancestor's
        result.
    mark_as : str, default "lineage_affine"
        Output column name.

    Raises
    ------
    NotImplementedError
        If `phylogeny_df` is not asexual, has non-contiguous ids, is not
        topologically sorted, or operands contain NaN or null values.

    See Also
    --------
    alifestd_mark_lineage_affine_asexual :
        Pandas-based implementation.
    alifestd_mark_lineage_cumsum_polars :
        Cumulative sum along each lineage.
    """
    schema_names = phylogeny_df.lazy().collect_schema().names()
    exprs = []
    for operand in (scale, offset):
        if isinstance(operand, str):
            if operand not in schema_names:
                raise ValueError(
                    f"column {operand!r} not found in phylogeny_df",
                )
            operand = pl.col(operand)
        exprs.append(operand.cast(pl.Float64))
    scale_expr, offset_expr = exprs

    phylogeny_df = alifestd_try_add_ancestor_id_col_polars(phylogeny_df)
    schema_names = phylogeny_df.lazy().collect_schema().names()
    if "ancestor_id" not in schema_names:
        raise NotImplementedError(
            "alifestd_mark_lineage_affine_polars only supports asexual "
            "phylogenies.",
        )

    if phylogeny_df.lazy().limit(1).collect().is_empty():
        return phylogeny_df.with_columns(offset_expr.alias(mark_as))

    if not alifestd_has_contiguous_ids_polars(phylogeny_df):
        raise NotImplementedError(
            "non-contiguous ids not supported",
        )

    if not alifestd_is_topologically_sorted_polars(phylogeny_df):
        raise NotImplementedError(
            "non-topologically-sorted data not supported",
        )

    logging.info(
        "- alifestd_mark_lineage_affine_polars: extracting columns...",
    )
    arrays = (
        phylogeny_df.lazy()
        .select(
            pl.col("ancestor_id"),
            scale_expr.alias("scale"),
            offset_expr.alias("offset"),
        )
        .collect()
    )

    logging.info(
        "- alifestd_mark_lineage_affine_polars: evaluating recurrence...",
    )
    scale_arr = arrays["scale"].fill_null(np.nan).to_numpy()
    offset_arr = arrays["offset"].fill_null(np.nan).to_numpy()
    if np.isnan(scale_arr).any() or np.isnan(offset_arr).any():
        raise NotImplementedError("NaN or null values not yet supported")

    result = _alifestd_mark_lineage_affine_asexual_fast_path(
        arrays["ancestor_id"].to_numpy(),
        scale_arr,
        offset_arr,
    )

    return phylogeny_df.with_columns(
        pl.Series(name=mark_as, values=result),
    )


_raw_description = f"""\
{os.path.basename(__file__)} | \
(phyloframe v{get_phyloframe_version()}/joinem v{joinem.__version__})

Add column evaluating affine recurrence
`scale * result[ancestor] + offset` down each lineage
(`offset` at roots).

Data is assumed to be in alife standard format.

Additional Notes
================
- Use `--eager-read` if modifying data file inplace.

- This CLI entrypoint is experimental and may be subject to change.

See Also
========
phyloframe.legacy._alifestd_mark_lineage_affine_asexual :
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
        dfcli_module="phyloframe.legacy._alifestd_mark_lineage_affine_polars",
        dfcli_version=get_phyloframe_version(),
    )
    parser.add_argument(
        "--scale",
        default=None,
        type=str,
        help="name of column multiplying ancestor's result",
    )
    parser.add_argument(
        "--offset",
        default=None,
        type=str,
        help="name of column added to multiplied ancestor's result",
    )
    parser.add_argument(
        "--mark-as",
        default="lineage_affine",
        type=str,
        help="output column name (default: lineage_affine)",
    )
    return parser


if __name__ == "__main__":
    begin_prod_logging()

    parser = _create_parser()
    args, __ = parser.parse_known_args()

    try:
        with log_context_duration(
            "phyloframe.legacy._alifestd_mark_lineage_affine_polars",
            logging.info,
        ):
            _run_dataframe_cli(
                base_parser=parser,
                output_dataframe_op=functools.partial(
                    alifestd_mark_lineage_affine_polars,
                    scale=args.scale,
                    offset=args.offset,
                    mark_as=args.mark_as,
                ),
            )
    except NotImplementedError as e:
        logging.error(
            "- polars op not yet implemented, use pandas op CLI instead",
        )
        raise e
