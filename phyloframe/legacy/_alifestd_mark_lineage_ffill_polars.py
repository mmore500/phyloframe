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
from ._alifestd_is_topologically_sorted_polars import (
    alifestd_is_topologically_sorted_polars,
)
from ._alifestd_mark_lineage_ffill_asexual import (
    _alifestd_mark_lineage_ffill_asexual_fast_path,
)
from ._alifestd_try_add_ancestor_id_col_polars import (
    alifestd_try_add_ancestor_id_col_polars,
)


def alifestd_mark_lineage_ffill_polars(
    phylogeny_df: pl.DataFrame,
    values: typing.Union[str, pl.Expr],
    *,
    mark_as: str = "lineage_ffill",
) -> pl.DataFrame:
    """Add column with null ``values`` filled from nearest ancestor with a
    non-null value.

    Non-null values are unchanged. Null values with no non-null ancestor
    (e.g., null roots) remain null. As with `polars.Expr.forward_fill`, only
    nulls are filled; NaN values are treated as values. Column dtype is
    preserved.

    Parameters
    ----------
    phylogeny_df : polars.DataFrame or polars.LazyFrame
        The phylogeny as a dataframe in alife standard format.

        Must represent an asexual phylogeny with contiguous ids and
        topologically sorted rows.
    values : str or polars.Expr
        Column name or polars expression providing per-node values.
    mark_as : str, default "lineage_ffill"
        Output column name.

    Returns
    -------
    polars.DataFrame or polars.LazyFrame
        The phylogeny with an added mark column.

    Raises
    ------
    NotImplementedError
        If `phylogeny_df` is not asexual, has non-contiguous ids, or is not
        topologically sorted.

    See Also
    --------
    alifestd_mark_lineage_ffill_asexual :
        Pandas-based implementation.
    alifestd_mark_lineage_cummax_polars :
        Maximum along lineages.
    """
    values_expr = pl.col(values) if isinstance(values, str) else values

    phylogeny_df = alifestd_try_add_ancestor_id_col_polars(phylogeny_df)
    if "ancestor_id" not in phylogeny_df.lazy().collect_schema().names():
        raise NotImplementedError(
            "alifestd_mark_lineage_ffill_polars only supports asexual "
            "phylogenies.",
        )

    if phylogeny_df.lazy().limit(1).collect().is_empty():
        return phylogeny_df.with_columns(values_expr.alias(mark_as))

    if not alifestd_has_contiguous_ids_polars(phylogeny_df):
        raise NotImplementedError("non-contiguous ids not supported")

    if not alifestd_is_topologically_sorted_polars(phylogeny_df):
        raise NotImplementedError(
            "non-topologically-sorted data not supported",
        )

    logging.info(
        "- alifestd_mark_lineage_ffill_polars: extracting columns...",
    )
    columns = (
        phylogeny_df.lazy()
        .select("ancestor_id", values_expr.alias(mark_as))
        .collect()
    )

    logging.info("- alifestd_mark_lineage_ffill_polars: filling values...")
    sources = _alifestd_mark_lineage_ffill_asexual_fast_path(
        columns["ancestor_id"].to_numpy(),
        columns[mark_as].is_null().to_numpy(),
    )

    return phylogeny_df.with_columns(columns[mark_as].gather(sources))


_raw_description = f"""{os.path.basename(__file__)} | (phyloframe v{get_phyloframe_version()}/joinem v{joinem.__version__})

Add column with null `values` filled from nearest ancestor with a non-null value.

Data is assumed to be in alife standard format.

Additional Notes
================
- Requires asexual phylogeny with contiguous ids and topologically sorted rows.

- Null values with no non-null ancestor (e.g., null roots) remain null.

- Use `--eager-read` if modifying data file inplace.

- This CLI entrypoint is experimental and may be subject to change.

See Also
========
phyloframe.legacy._alifestd_mark_lineage_ffill_asexual :
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
        dfcli_module="phyloframe.legacy._alifestd_mark_lineage_ffill_polars",
        dfcli_version=get_phyloframe_version(),
    )
    parser.add_argument(
        "--values",
        default=None,
        type=str,
        help="name of column to fill along lineage",
    )
    parser.add_argument(
        "--mark-as",
        default="lineage_ffill",
        type=str,
        help="output column name (default: lineage_ffill)",
    )
    return parser


if __name__ == "__main__":
    begin_prod_logging()

    parser = _create_parser()
    args, __ = parser.parse_known_args()

    try:
        with log_context_duration(
            "phyloframe.legacy._alifestd_mark_lineage_ffill_polars",
            logging.info,
        ):
            _run_dataframe_cli(
                base_parser=parser,
                output_dataframe_op=functools.partial(
                    alifestd_mark_lineage_ffill_polars,
                    values=args.values,
                    mark_as=args.mark_as,
                ),
            )
    except NotImplementedError as e:
        logging.error(
            "- polars op not yet implemented, use pandas op CLI instead",
        )
        raise e
