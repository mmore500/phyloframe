import argparse
import functools
import logging
import os
import typing

import joinem
from joinem._dataframe_cli import _add_parser_base, _run_dataframe_cli
import numpy as np
import polars as pl
import polars.selectors as cs

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
) -> typing.Tuple[np.ndarray, int]:
    """Implementation detail for `alifestd_mark_joint_quantile_bins_polars`.

    Assigns items to equal-occupancy bins, the leaves of a balanced k-d tree
    over items' tiebroken ranks (see `calc_tiebroken_ranks`). Uses the
    largest power of two of bins, up to `max_num_bins`, that keeps at least
    `min_bin_occupancy` items in each; splitting can stop early, giving
    fewer bins, where items do not divide evenly.

    Returns each item's bin, numbered contiguously from 0, and the number of
    bins.
    """
    assert int(max_num_bins).bit_count() == 1
    assert min_bin_occupancy >= 1
    num_items = len(coordinates[0])
    if num_items == 0:
        return np.zeros(0, dtype=np.int64), 0

    ranks = calc_tiebroken_ranks(coordinates) / num_items  # scaled to (0, 1]
    num_levels = int(
        np.clip(
            np.log2(num_items / min_bin_occupancy), 0, np.log2(max_num_bins)
        )
    )
    # splitting stops at most `num_levels` levels down, as nodes there hold
    # at most this many items, but can stop early, as nodes above may too
    bins = assign_kdtree_partitions(
        ranks, ceil_div(num_items, 2**num_levels)
    )
    return bins, int(bins.max()) + 1


def alifestd_mark_joint_quantile_bins_polars(
    phylogeny_df: pl.DataFrame,
    *,
    bin_by: typing.Sequence[typing.Union[str, pl.Expr]],
    partition_by: typing.Union[str, pl.Expr, None] = None,
    max_num_bins: int = 4096,
    min_bin_occupancy: int = 4,
    mark_as: str = "joint_quantile_bin",
) -> pl.DataFrame:
    """Add column `joint_quantile_bin`, assigning rows to equal-occupancy
    bins by their joint quantiles over columns `bin_by`, separately within
    each value of column `partition_by`, if given.

    Bins are the leaves of a balanced k-d tree over rows' ranks, which
    repeatedly splits rows at the median of one of the columns. Ties on one
    column are broken by the others, so bins do not depend on row order
    unless rows tie on every column. Each partition gets the largest power
    of two of bins, up to `max_num_bins`, that keeps at least
    `min_bin_occupancy` rows in each, or fewer where rows do not divide
    evenly.
    Bins are numbered consecutively across partitions, without gaps, in
    order of partition value, so each bin lies within one partition.

    Columns constant within a partition are skipped there: their ranks,
    tiebroken by the other columns, would only repeat those columns'
    splits.

    The output column name can be changed via the ``mark_as`` parameter.

    Parameters
    ----------
    phylogeny_df : polars.DataFrame or polars.LazyFrame
        The phylogeny as a dataframe in alife standard format (or any
        dataframe).
    bin_by : Sequence[str or polars.Expr]
        Numeric columns, by name or expression, to bin rows by, e.g., branch
        length and origin time.
    partition_by : str or polars.Expr, optional
        Column, by name or expression, whose values group rows to bin
        separately, e.g., from `alifestd_mark_clade_size_classes_polars`.
    max_num_bins : int, default 4096
        Largest number of bins per partition; a power of two.
    min_bin_occupancy : int, default 4
        Fewest rows per bin, unless a partition has fewer rows in all.
    mark_as : str, default "joint_quantile_bin"
        Output column name.

    Returns
    -------
    polars.DataFrame
        The phylogeny with an added integer column of bins.

    Raises
    ------
    NotImplementedError
        If a `bin_by` or `partition_by` value is null or nan.
    ValueError
        If `bin_by` is empty, if `max_num_bins` is not a power of two, or if
        `min_bin_occupancy` is less than 1.

    Notes
    -----
    For calibration bins of `alifestd_tgs_estimate_polars`, see its Notes.

    See Also
    --------
    alifestd_mark_clade_size_classes_polars :
        Marks clade size classes, e.g., as partitions.
    """
    if isinstance(bin_by, str):
        raise TypeError(
            "bin_by must be a sequence of column names, not a single "
            f"string; did you mean [{bin_by!r}]?",
        )
    if not len(bin_by):
        raise ValueError("bin_by must be nonempty")
    if int(max_num_bins).bit_count() != 1 or max_num_bins < 1:
        raise ValueError(
            f"max_num_bins must be a power of two, got {max_num_bins}"
        )
    if min_bin_occupancy < 1:
        raise ValueError(
            f"min_bin_occupancy must be at least 1, got {min_bin_occupancy}"
        )

    def as_expr(column: typing.Union[str, pl.Expr]) -> pl.Expr:
        return pl.col(column) if isinstance(column, str) else column

    selected = (
        phylogeny_df.lazy()
        .select(
            *(
                as_expr(c).cast(pl.Float64).alias(f"bin_by{i}")
                for i, c in enumerate(bin_by)
            ),
            (
                pl.lit(0) if partition_by is None else as_expr(partition_by)
            ).alias("partition"),
        )
        .with_row_index("row")
        .collect()
    )
    if selected.select(
        pl.any_horizontal(pl.all().is_null(), cs.float().is_nan()).any()
    ).item():
        raise NotImplementedError(
            "null or nan bin_by or partition_by values not supported",
        )
    marks = np.empty(len(selected), dtype=np.int64)
    offset = 0  # bins are numbered consecutively across partitions
    # stable sort keeps row order within partitions, which breaks full ties
    for (partition,), group in selected.sort(
        "partition", maintain_order=True
    ).group_by("partition", maintain_order=True):
        coordinates = [
            group[f"bin_by{i}"].to_numpy() for i in range(len(bin_by))
        ]
        bins, num_bins = _alifestd_mark_joint_quantile_bins_fast_path(
            [x for x in coordinates if np.ptp(x) > 0] or coordinates,
            max_num_bins=max_num_bins,
            min_bin_occupancy=min_bin_occupancy,
        )
        marks[group["row"].to_numpy()] = bins + offset
        offset += num_bins
        logging.info(
            "- alifestd_mark_joint_quantile_bins_polars: partition "
            f"{partition} has {len(group)} rows in {num_bins} bins",
        )
    return phylogeny_df.with_columns(
        pl.Series(name=mark_as, values=marks, dtype=pl.Int64),
    )


_raw_description = f"""\
{os.path.basename(__file__)} | \
(phyloframe v{get_phyloframe_version()}/joinem v{joinem.__version__})

Add column `joint_quantile_bin`, assigning rows to equal-occupancy bins by
their joint quantiles over columns `--bin-by`, separately within each value of
`--partition-by`, if given.

Additional Notes
================
- Bins are numbered consecutively across partitions.

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
