import argparse
import functools
import logging
import os
import typing

import joinem
from joinem._dataframe_cli import _add_parser_base, _run_dataframe_cli
import pandas as pd

from .._auxlib._begin_prod_logging import begin_prod_logging
from .._auxlib._delegate_polars_implementation import (
    delegate_polars_implementation,
)
from .._auxlib._format_cli_description import format_cli_description
from .._auxlib._get_phyloframe_version import get_phyloframe_version
from .._auxlib._log_context_duration import log_context_duration
from ._alifestd_aggregate_phylogenies import alifestd_aggregate_phylogenies


def alifestd_aggregate_phylogeny_partitions(
    phylogeny_df: pd.DataFrame,
    partition_by: typing.Union[str, typing.Sequence[str]],
    mutate: bool = False,
) -> pd.DataFrame:
    """Split a dataframe into partitions by the values of one or more columns,
    then concatenate the partitions as independent phylogenies, reassigning
    organism ids to prevent collisions.

    Partitions are ordered by first appearance of their key in
    `phylogeny_df`, and rows retain their relative order within each
    partition. Each partition is expected to be a self-contained phylogeny,
    i.e., no organism should have an ancestor in a different partition.

    Input dataframe is not mutated by this operation unless `mutate` set True.
    If mutate set True, operation does not occur in place; still use return
    value to get transformed phylogeny dataframe.

    Parameters
    ----------
    phylogeny_df : pandas.DataFrame
        The phylogeny as a dataframe in alife standard format.
    partition_by : str or sequence of str
        Name(s) of the column(s) to split the dataframe over.
    mutate : bool, default False
        If True, allow mutation of the input dataframe.

    Returns
    -------
    pandas.DataFrame
        The aggregated phylogenies, with unique organism ids.

    See Also
    --------
    alifestd_aggregate_phylogeny_partitions_polars :
        Polars-based implementation.
    alifestd_aggregate_phylogenies :
        Aggregates a list of separate phylogeny dataframes.
    """
    if not mutate:
        phylogeny_df = phylogeny_df.copy()

    if len(phylogeny_df) == 0:
        return phylogeny_df

    partitions = [
        partition
        for __, partition in phylogeny_df.groupby(
            partition_by,
            sort=False,
            dropna=False,
        )
    ]
    return alifestd_aggregate_phylogenies(partitions, mutate=True)


_raw_description = f"""{os.path.basename(__file__)} | (phyloframe v{get_phyloframe_version()}/joinem v{joinem.__version__})

Split data into partitions by the values of `--partition-by` column(s),
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
phyloframe.legacy._alifestd_aggregate_phylogeny_partitions_polars :
    Entrypoint for high-performance Polars-based implementation.
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
        dfcli_module="phyloframe.legacy._alifestd_aggregate_phylogeny_partitions",
        dfcli_version=get_phyloframe_version(),
    )
    parser.add_argument(
        "--partition-by",
        action="append",
        dest="partition_by",
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
    with log_context_duration(
        "phyloframe.legacy._alifestd_aggregate_phylogeny_partitions",
        logging.info,
    ):
        _run_dataframe_cli(
            base_parser=parser,
            output_dataframe_op=delegate_polars_implementation()(
                functools.partial(
                    alifestd_aggregate_phylogeny_partitions,
                    partition_by=args.partition_by,
                ),
            ),
        )
