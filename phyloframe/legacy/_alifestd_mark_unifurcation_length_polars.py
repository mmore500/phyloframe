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
from ._alifestd_mark_unifurcation_length_asexual import (
    _alifestd_mark_unifurcation_length_asexual_fast_path,
)
from ._alifestd_try_add_ancestor_id_col_polars import (
    alifestd_try_add_ancestor_id_col_polars,
)


def alifestd_mark_unifurcation_length_polars(
    phylogeny_df: pl.DataFrame,
    *,
    mark_as: str = "unifurcation_length",
    weights: typing.Optional[typing.Union[str, pl.Expr]] = None,
) -> pl.DataFrame:
    """Add column `unifurcation_length`, measuring for each node the run of
    contiguous unifurcations (nodes with exactly one child) directly above
    it, i.e., the ancestors that `alifestd_collapse_unifurcations_polars`
    would bypass.

    By default, length counts unifurcating nodes. If ``weights`` is
    provided (column name or expression), length sums its values over
    those nodes instead (e.g., to measure length in branch length units,
    or to accumulate counts across repeated unifurcation collapses).

    As in `alifestd_collapse_unifurcations_polars`, roots are not considered
    unifurcations. The output column name can be changed via the
    ``mark_as`` parameter.

    Raises
    ------
    NotImplementedError
        If `phylogeny_df` is not asexual, has non-contiguous ids, or is not
        topologically sorted.

    See Also
    --------
    alifestd_mark_unifurcation_length_asexual :
        Pandas-based implementation.
    alifestd_collapse_unifurcations_polars :
        Removes the unifurcations measured here.
    """
    schema_names = phylogeny_df.lazy().collect_schema().names()
    if isinstance(weights, str):
        if weights not in schema_names:
            raise ValueError(
                f"weights column {weights!r} not found in phylogeny_df",
            )
        weights = pl.col(weights)

    phylogeny_df = alifestd_try_add_ancestor_id_col_polars(phylogeny_df)
    schema_names = phylogeny_df.lazy().collect_schema().names()
    if "ancestor_id" not in schema_names:
        raise NotImplementedError(
            "alifestd_mark_unifurcation_length_polars only supports asexual "
            "phylogenies.",
        )

    if phylogeny_df.lazy().limit(1).collect().is_empty():
        return phylogeny_df.with_columns(
            (pl.lit(0) if weights is None else weights).alias(mark_as),
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
        "- alifestd_mark_unifurcation_length_polars: "
        "extracting ancestor ids...",
    )
    ancestor_ids = (
        phylogeny_df.lazy()
        .select("ancestor_id")
        .collect()
        .to_series()
        .to_numpy()
    )
    weights_arr = (
        None
        if weights is None
        else phylogeny_df.lazy().select(weights).collect().to_series()
    )
    if weights_arr is not None:
        weights_arr = weights_arr.to_numpy()

    logging.info(
        "- alifestd_mark_unifurcation_length_polars: "
        "measuring unifurcations...",
    )
    result = _alifestd_mark_unifurcation_length_asexual_fast_path(
        ancestor_ids,
        weights_arr,
    )

    return phylogeny_df.with_columns(
        pl.Series(name=mark_as, values=result),
    )


_raw_description = f"""\
{os.path.basename(__file__)} | \
(phyloframe v{get_phyloframe_version()}/joinem v{joinem.__version__})

Add column `unifurcation_length`, measuring for each node the run of
contiguous unifurcations directly above it (i.e., ancestors that
unifurcation collapse would bypass).

Data is assumed to be in alife standard format.

Additional Notes
================
- Use `--weights` to sum a column over unifurcations, instead of counting
them.

- Use `--eager-read` if modifying data file inplace.

- This CLI entrypoint is experimental and may be subject to change.

See Also
========
phyloframe.legacy._alifestd_mark_unifurcation_length_asexual :
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
        dfcli_module="phyloframe.legacy._alifestd_mark_unifurcation_length_polars",
        dfcli_version=get_phyloframe_version(),
    )
    parser.add_argument(
        "--mark-as",
        default="unifurcation_length",
        type=str,
        help="output column name (default: unifurcation_length)",
    )
    parser.add_argument(
        "--weights",
        default=None,
        type=str,
        help="column to sum over unifurcations (default: None, count them)",
    )
    return parser


if __name__ == "__main__":
    begin_prod_logging()

    parser = _create_parser()
    args, __ = parser.parse_known_args()

    try:
        with log_context_duration(
            "phyloframe.legacy._alifestd_mark_unifurcation_length_polars",
            logging.info,
        ):
            _run_dataframe_cli(
                base_parser=parser,
                output_dataframe_op=functools.partial(
                    alifestd_mark_unifurcation_length_polars,
                    mark_as=args.mark_as,
                    weights=args.weights,
                ),
            )
    except NotImplementedError as e:
        logging.error(
            "- polars op not yet implemented, use pandas op CLI instead",
        )
        raise e
