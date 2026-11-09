import argparse
import functools
import logging
import os
import typing

import joinem
from joinem._dataframe_cli import _add_parser_base, _run_dataframe_cli
import numpy as np
import polars as pl

from .._auxlib._assign_kdtree_partitions import assign_kdtree_partitions
from .._auxlib._begin_prod_logging import begin_prod_logging
from .._auxlib._calc_tiebroken_ranks import calc_tiebroken_ranks
from .._auxlib._ceil_div import ceil_div
from .._auxlib._format_cli_description import format_cli_description
from .._auxlib._get_phyloframe_version import get_phyloframe_version
from .._auxlib._log_context_duration import log_context_duration


def _alifestd_mark_joint_quantile_bins_fast_path(
    coordinates: typing.Sequence[np.ndarray],
    *,
    max_num_bins: int,
    min_bin_occupancy: int,
) -> np.ndarray:
    """Implementation detail for `alifestd_mark_joint_quantile_bins_polars`.

    Returns each item's bin, numbered from 0.
    """
    num_items = len(coordinates[0])
    # largest power of two that keeps bins at least min_bin_occupancy
    num_bins = 2 ** int(
        np.log2(np.clip(num_items // min_bin_occupancy, 1, max_num_bins))
    )
    return assign_kdtree_partitions(
        calc_tiebroken_ranks(coordinates),
        leafsize=ceil_div(num_items, num_bins),
    )


def alifestd_mark_joint_quantile_bins_polars(
    phylogeny_df: pl.DataFrame,
    *,
    bin_by: typing.Union[str, typing.Sequence[str]],
    partition_by: typing.Optional[str] = None,
    max_num_bins: int = 4096,
    min_bin_occupancy: int = 4,
    mark_as: str = "joint_quantile_bin",
) -> pl.DataFrame:
    """Add column `joint_quantile_bin`, assigning rows to equal-occupancy
    bins by their joint quantiles over columns `bin_by`.

    Bins are the leaves of a balanced k-d tree over rows' ranks, with ties
    on one column broken by the others. If `partition_by` is given, rows are
    binned separately within each of its values, with bins numbered
    consecutively across partitions.

    The output column name can be changed via the ``mark_as`` parameter.

    Parameters
    ----------
    phylogeny_df : polars.DataFrame or polars.LazyFrame
        The phylogeny as a dataframe in alife standard format (or any
        dataframe).
    bin_by : str or Sequence[str]
        Numeric columns to bin rows by.
    partition_by : str, optional
        Column whose values group rows to bin separately.
    max_num_bins : int, default 4096
        Most bins per partition; rounded down to a power of two.
    min_bin_occupancy : int, default 4
        Fewest rows per bin, unless a partition has fewer rows in all.
    mark_as : str, default "joint_quantile_bin"
        Output column name.

    See Also
    --------
    alifestd_mark_clade_size_geom_classes_polars :
        Marks clade size classes, e.g., as partitions.
    """
    if isinstance(bin_by, str):
        bin_by = [bin_by]
    partition = pl.lit(0) if partition_by is None else pl.col(partition_by)

    if (
        phylogeny_df.lazy()
        .select(
            pl.any_horizontal(
                pl.col(*bin_by).is_nan(),
                pl.col(*bin_by).is_null(),
                partition.is_null(),
            ).any(),
        )
        .collect()
        .item()
    ):
        raise NotImplementedError("null or nan values not supported")

    logging.info(
        "- alifestd_mark_joint_quantile_bins_polars: binning...",
    )
    return phylogeny_df.with_columns(
        pl.struct(bin_by)
        .map_batches(
            lambda rows: pl.Series(
                _alifestd_mark_joint_quantile_bins_fast_path(
                    [col.to_numpy() for col in rows.struct.unnest()],
                    max_num_bins=max_num_bins,
                    min_bin_occupancy=min_bin_occupancy,
                ),
            ),
            return_dtype=pl.Int64,
        )
        .over(partition)
        .alias(mark_as),
    ).with_columns(  # number bins consecutively across partitions
        (pl.struct(partition, mark_as).rank("dense") - 1)
        .cast(pl.Int64)
        .alias(mark_as),
    )


_raw_description = f"""{os.path.basename(__file__)} | (phyloframe v{get_phyloframe_version()}/joinem v{joinem.__version__})

Add column `joint_quantile_bin`, assigning rows to equal-occupancy bins by their joint quantiles over columns `--bin-by`.

Additional Notes
================
- Use `--eager-read` if modifying data file inplace.

- This CLI entrypoint is experimental and may be subject to change.
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
        dfcli_module="phyloframe.legacy._alifestd_mark_joint_quantile_bins_polars",
        dfcli_version=get_phyloframe_version(),
    )
    parser.add_argument(
        "--bin-by",
        nargs="+",
        required=True,
        type=str,
        help="Columns to bin rows by.",
    )
    parser.add_argument(
        "--partition-by",
        default=None,
        type=str,
        help="Column of partitions to bin separately (default: none).",
    )
    parser.add_argument(
        "--max-num-bins",
        default=4096,
        type=int,
        help="Largest number of bins per partition (default: 4096).",
    )
    parser.add_argument(
        "--min-bin-occupancy",
        default=4,
        type=int,
        help="Fewest rows per bin (default: 4).",
    )
    parser.add_argument(
        "--mark-as",
        default="joint_quantile_bin",
        type=str,
        help="Output column name (default: joint_quantile_bin).",
    )
    return parser


if __name__ == "__main__":
    begin_prod_logging()

    parser = _create_parser()
    args, __ = parser.parse_known_args()

    with log_context_duration(
        "phyloframe.legacy._alifestd_mark_joint_quantile_bins_polars",
        logging.info,
    ):
        _run_dataframe_cli(
            base_parser=parser,
            output_dataframe_op=functools.partial(
                alifestd_mark_joint_quantile_bins_polars,
                bin_by=args.bin_by,
                partition_by=args.partition_by,
                max_num_bins=args.max_num_bins,
                min_bin_occupancy=args.min_bin_occupancy,
                mark_as=args.mark_as,
            ),
        )
