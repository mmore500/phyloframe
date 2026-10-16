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
from ._alifestd_aggregate_phylogenies_polars import (
    alifestd_aggregate_phylogenies_polars,
)


def alifestd_aggregate_phylogeny_partitions_polars(
    phylogeny_df: typing.Union[pl.DataFrame, pl.LazyFrame],
    partitioned_by: typing.Union[
        str,
        pl.Expr,
        typing.Sequence[typing.Union[str, pl.Expr]],
    ],
    *more_partitioned_by: typing.Union[str, pl.Expr],
) -> pl.DataFrame:
    """Split a dataframe into partitions by the values of one or more columns
    or expressions, then concatenate the partitions as independent
    phylogenies, reassigning organism ids to prevent collisions.

    Partitions are ordered by first appearance of their key in
    `phylogeny_df`, and rows retain their relative order within each
    partition. Each partition is expected to be a self-contained phylogeny,
    i.e., no organism should have an ancestor in a different partition.

    Delegates to ``alifestd_aggregate_phylogenies_polars``, so each partition
    must have contiguous ids and be topologically sorted.

    Parameters
    ----------
    phylogeny_df : polars.DataFrame or polars.LazyFrame
        The phylogeny as a dataframe in alife standard format.

        Must represent an asexual phylogeny with an ``ancestor_id`` column
        (not ``ancestor_list``).
    partitioned_by : str, polars.Expr, or sequence of str and polars.Expr
        Column name(s), selector(s), or expression(s) to split the dataframe
        over.
    *more_partitioned_by : str or polars.Expr
        Additional column names, selectors, or expressions to split over,
        specified as positional arguments.

    Returns
    -------
    polars.DataFrame
        The aggregated phylogenies, with unique organism ids.

    See Also
    --------
    alifestd_aggregate_phylogeny_partitions :
        Pandas-based implementation.
    alifestd_aggregate_phylogenies_polars :
        Aggregates a list of separate phylogeny dataframes.
    """
    if isinstance(partitioned_by, (str, pl.Expr)):
        partitioned_by = [partitioned_by]

    phylogeny_df = (
        phylogeny_df.lazy()
        .with_columns(
            pl.struct(*partitioned_by, *more_partitioned_by).alias(
                "_partition",
            ),
        )
        .collect()
    )
    if phylogeny_df.is_empty():
        return phylogeny_df.drop("_partition")

    return alifestd_aggregate_phylogenies_polars(
        phylogeny_df.partition_by(
            "_partition",
            maintain_order=True,
            include_key=False,
        ),
    )


_raw_description = f"""{os.path.basename(__file__)} | (phyloframe v{get_phyloframe_version()}/joinem v{joinem.__version__})

Split data into partitions by the values of `--partitioned-by` column(s),
then concatenate the partitions as independent phylogenies, reassigning
organism ids to prevent collisions.

Data is assumed to be in alife standard format.

Additional Notes
================
- Use `--with-column` to add a column to partition by, e.g., to partition by
  source file, `--with-column 'pl.lit(filepath).alias("source")'`.

- Use `--eager-read` if modifying data file inplace.

- This CLI entrypoint is experimental and may be subject to change.

See Also
========
phyloframe.legacy._alifestd_aggregate_phylogeny_partitions :
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
        dfcli_module="phyloframe.legacy._alifestd_aggregate_phylogeny_partitions_polars",
        dfcli_version=get_phyloframe_version(),
    )
    parser.add_argument(
        "--partitioned-by",
        action="append",
        dest="partitioned_by",
        type=str,
        default=[],
        help=(
            "Name of column to split data over. "
            "Specify multiple columns by using this flag multiple times."
        ),
    )
    return parser


if __name__ == "__main__":
    begin_prod_logging()

    parser = _create_parser()
    args, __ = parser.parse_known_args()

    try:
        with log_context_duration(
            "phyloframe.legacy._alifestd_aggregate_phylogeny_partitions_polars",
            logging.info,
        ):
            _run_dataframe_cli(
                base_parser=parser,
                output_dataframe_op=functools.partial(
                    alifestd_aggregate_phylogeny_partitions_polars,
                    partitioned_by=args.partitioned_by,
                ),
            )
    except NotImplementedError as e:
        logging.error(
            "- polars op not yet implemented, use pandas op CLI instead",
        )
        raise e
