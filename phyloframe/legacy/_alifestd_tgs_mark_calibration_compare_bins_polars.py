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
from .._auxlib._calc_geometric_bins import calc_geometric_bins
from .._auxlib._format_cli_description import format_cli_description
from .._auxlib._get_phyloframe_version import get_phyloframe_version
from .._auxlib._log_context_duration import log_context_duration
from ._impl._tgs_build_exposure_df_polars import tgs_build_exposure_df_polars
from ._impl._tgs_partition_by_rank import tgs_partition_by_rank


def _alifestd_tgs_mark_size_partitions_fast_path(
    *,
    clade_sizes: np.ndarray,
    neutral_gain_positions: np.ndarray,
    ratio: float,
    min_neutral_gains: int,
) -> np.ndarray:
    """Implementation detail for
    `alifestd_tgs_mark_calibration_compare_bins_polars`. Acronym tgs stands for
    trait gain/selection.

    Returns each branch's clade size partition, given its clade size
    (number of leaves below it) and the positions of branches that gained a
    neutral trait.
    """
    clade_sizes = np.asarray(clade_sizes)
    # number occupied bins consecutively, dropping empty ones
    __, partitions = np.unique(
        calc_geometric_bins(clade_sizes, ratio=ratio), return_inverse=True
    )
    num_partitions = partitions.max(initial=0) + 1
    neutral_gains = np.bincount(
        partitions[neutral_gain_positions], minlength=num_partitions
    )
    # partitions 1 and up hold inner nodes; leaves keep partition 0
    (sparse_inner,) = np.nonzero(neutral_gains[1:] < min_neutral_gains)
    if len(sparse_inner):
        cap = max(sparse_inner[0], 1)
        if cap < num_partitions - 1:
            logging.info(
                "- alifestd_tgs_mark_calibration_compare_bins_polars: clade "
                f"size partitions above {cap} have fewer than "
                f"{min_neutral_gains} neutral gains, so join partition {cap}",
            )
        partitions = np.minimum(partitions, cap)
        num_partitions = cap + 1
        neutral_gains = np.bincount(
            partitions[neutral_gain_positions], minlength=num_partitions
        )

    for partition in range(num_partitions):
        sizes = clade_sizes[partitions == partition]
        logging.info(
            "- alifestd_tgs_mark_calibration_compare_bins_polars: clade size "
            f"partition {partition} has clade sizes "
            f"{sizes.min()}-{sizes.max()}, {len(sizes)} branches, and "
            f"{neutral_gains[partition]} neutral trait gains",
        )
    return partitions


def alifestd_tgs_mark_calibration_compare_bins_polars(
    phylogeny_df: pl.DataFrame,
    *,
    neutral_traits: typing.Sequence[str],
    max_bins: int = 4096,
    size_ratio: float = 2.0,
    min_neutral_gains: int = 20,
    mark_partition_as: str = "tgs_calibration_partition",
    mark_bin_as: str = "tgs_calibration_compare_bin",
) -> pl.DataFrame:
    """Mark each node's calibration partition, by clade size, and its
    calibration bin within it, by branch length and origin time, for the
    `calibration_partition_by` and `calibration_compare_by_bins` of
    `alifestd_tgs_pendantmdc_estimate_polars` and
    `alifestd_tgs_allmdc_estimate_polars`.

    Partitions and bins are computed on the phylogeny with unifurcations
    collapsed, as the estimators fit it, and are marked at each branch's end
    node. Roots and collapsed unifurcations, which end no branch, are marked
    -1.

    First, branches are partitioned by clade size, the number of leaves
    below, on a log scale: a branch above `n` leaves falls in size class
    ceil(log(n) / log(`size_ratio`)), so the default ratio of 2 gives size
    classes of 1, 2, 3-4, 5-8, 9-16, ... leaves. Size classes absent from
    the tree are skipped, so leaf branches are always partition 0. So that every partition has neutral
    trait gains to calibrate against, partitions are then merged from the
    largest clades down: the first partition above the leaves' with fewer
    than `min_neutral_gains` gains of `neutral_traits`, and every partition
    of larger clades, joins the partition below it. If even the first inner
    partition has too few gains, all inner branches form one partition.
    Leaf branches always keep their own partition.

    Then, each partition's branches are split into up to `max_bins`
    equal-occupancy bins by ranked branch length and origin time, keeping
    at least 4 branches in each. Bins are numbered consecutively across
    partitions.

    Parameters
    ----------
    phylogeny_df : polars.DataFrame or polars.LazyFrame
        The phylogeny as a dataframe in alife standard format, with column
        `origin_time`.

        Must represent an asexual phylogeny with contiguous,
        topologically sorted ids.
    neutral_traits : Sequence[str]
        Names of integer columns giving each neutral trait's state
        transition on each node's branch, as for the estimators'
        `trait_transitions`.
    max_bins : int, default 4096
        Maximum number of bins per partition; a power of two.
    size_ratio : float, default 2.0
        Ratio between successive partitions' clade size bounds; must exceed
        1. Larger ratios give fewer, larger partitions.
    min_neutral_gains : int, default 20
        Fewest neutral trait gains a partition of inner branches may hold
        before merging; 0 disables merging.
    mark_partition_as : str, default "tgs_calibration_partition"
        Output column name for partitions.
    mark_bin_as : str, default "tgs_calibration_compare_bin"
        Output column name for bins.

    Returns
    -------
    polars.DataFrame
        The phylogeny with added integer columns of calibration partitions
        and bins.

    Raises
    ------
    NotImplementedError
        If `phylogeny_df` is not asexual, or its ids are non-contiguous or
        not topologically sorted.
    ValueError
        If a `neutral_traits` column is missing or gives states other than 0
        and 1, if `max_bins` is not a power of two, if `size_ratio`
        does not exceed 1, or if `min_neutral_gains` is negative.

    See Also
    --------
    alifestd_tgs_pendantmdc_estimate_polars :
        Calibrates leaf pendant branch lengths by these bins.
    alifestd_tgs_allmdc_estimate_polars :
        Calibrates branch lengths by these bins.
    """
    if int(max_bins).bit_count() != 1 or max_bins < 1:
        raise ValueError(f"max_bins must be a power of two, got {max_bins}")
    if not size_ratio > 1:
        raise ValueError(f"size_ratio must exceed 1, got {size_ratio}")
    if min_neutral_gains < 0:
        raise ValueError(
            f"min_neutral_gains must be nonnegative, got {min_neutral_gains}",
        )
    if isinstance(neutral_traits, str):
        raise TypeError(
            "neutral_traits must be a sequence of column names, not a "
            f"single string; did you mean [{neutral_traits!r}]?",
        )

    neutral_traits = list(dict.fromkeys(neutral_traits))  # distinct, in order
    exposure_df = tgs_build_exposure_df_polars(
        phylogeny_df,
        trait_transitions=neutral_traits,
        caller="alifestd_tgs_mark_calibration_compare_bins_polars",
    )
    num_nodes = phylogeny_df.lazy().select(pl.len()).collect().item()
    partition_marks = np.full(num_nodes, -1, dtype=np.int64)
    bin_marks = partition_marks.copy()

    def mark() -> pl.DataFrame:
        return phylogeny_df.with_columns(
            pl.Series(name=mark_partition_as, values=partition_marks),
            pl.Series(name=mark_bin_as, values=bin_marks),
        )

    if exposure_df.empty:
        return mark()

    partitions = _alifestd_tgs_mark_size_partitions_fast_path(
        clade_sizes=exposure_df["clade_size"].to_numpy(),
        neutral_gain_positions=np.concatenate(
            [np.zeros(0, dtype=np.int64)]
            + [
                np.flatnonzero(exposure_df[f"trait{i}_gain"].to_numpy())
                for i in range(len(neutral_traits))
            ],
        ),
        ratio=size_ratio,
        min_neutral_gains=min_neutral_gains,
    )
    bins = np.empty(len(exposure_df), dtype=np.int64)
    offset = 0
    for partition in np.unique(partitions):
        (members,) = np.nonzero(partitions == partition)
        # not survival duration, which the allmdc model accounts for:
        # calibrating along it would absorb its effect on neutral gains
        # (e.g., from loss) into branch length, then count it again
        coordinates = [
            exposure_df[column].to_numpy()[members]
            for column in ("branch_length", "origin_time")
        ]
        member_bins, num, __ = tgs_partition_by_rank(
            # constant coordinates would only split on ties
            coordinates=[x for x in coordinates if np.ptp(x) > 0]
            or coordinates,
            max_partitions=max_bins,
        )
        bins[members] = member_bins + offset
        offset += num
        logging.info(
            "- alifestd_tgs_mark_calibration_compare_bins_polars: partition "
            f"{partition} has {len(members)} branches in {num} bins",
        )
    node_ids = exposure_df["node_id"].to_numpy()
    partition_marks[node_ids] = partitions
    bin_marks[node_ids] = bins
    return mark()


_raw_description = f"""\
{os.path.basename(__file__)} | \
(phyloframe v{get_phyloframe_version()}/joinem v{joinem.__version__})

Mark each node's calibration partition, by clade size on a log scale (merging
partitions with too few neutral trait gains), and its calibration bin within
it, by ranked branch length and origin time, for `--calibration-partition-by`
and `--calibration-compare-by-bins` of
`phyloframe.legacy._alifestd_tgs_pendantmdc_estimate_polars` and
`phyloframe.legacy._alifestd_tgs_allmdc_estimate_polars`.

Data is assumed to be in alife standard format, with column `origin_time`.
Only supports asexual phylogenies.

Additional Notes
================
- Requires contiguous, topologically sorted ids.

- Roots and collapsed unifurcations are marked -1.

- Output columns are tgs_calibration_partition and
  tgs_calibration_compare_bin, unless renamed.

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
        dfcli_module=(
            "phyloframe.legacy._alifestd_tgs_mark_calibration_compare_bins_polars"
        ),
        dfcli_version=get_phyloframe_version(),
    )
    parser.add_argument(
        "--neutral-traits",
        nargs="+",
        required=True,
        type=str,
        help="Names of neutral trait transition columns.",
    )
    parser.add_argument(
        "--max-bins",
        default=4096,
        type=int,
        help=(
            "Maximum number of bins per partition, a power of two "
            "(default: 4096)."
        ),
    )
    parser.add_argument(
        "--size-ratio",
        default=2.0,
        type=float,
        help=(
            "Ratio between successive clade size partitions (default: 2.0)."
        ),
    )
    parser.add_argument(
        "--min-neutral-gains",
        default=20,
        type=int,
        help=(
            "Fewest neutral trait gains an inner partition may hold before "
            "merging; 0 disables merging (default: 20)."
        ),
    )
    parser.add_argument(
        "--mark-partition-as",
        default="tgs_calibration_partition",
        type=str,
        help=(
            "Output column name for partitions (default: "
            "tgs_calibration_partition)."
        ),
    )
    parser.add_argument(
        "--mark-bin-as",
        default="tgs_calibration_compare_bin",
        type=str,
        help=(
            "Output column name for bins (default: "
            "tgs_calibration_compare_bin)."
        ),
    )
    return parser


if __name__ == "__main__":
    begin_prod_logging()

    parser = _create_parser()
    args, __ = parser.parse_known_args()

    try:
        with log_context_duration(
            "phyloframe.legacy._alifestd_tgs_mark_calibration_compare_bins_polars",
            logging.info,
        ):
            _run_dataframe_cli(
                base_parser=parser,
                output_dataframe_op=functools.partial(
                    alifestd_tgs_mark_calibration_compare_bins_polars,
                    neutral_traits=args.neutral_traits,
                    max_bins=args.max_bins,
                    size_ratio=args.size_ratio,
                    min_neutral_gains=args.min_neutral_gains,
                    mark_partition_as=args.mark_partition_as,
                    mark_bin_as=args.mark_bin_as,
                ),
            )
    except NotImplementedError as e:
        logging.error(
            "- polars op not yet implemented, use pandas op instead",
        )
        raise e
