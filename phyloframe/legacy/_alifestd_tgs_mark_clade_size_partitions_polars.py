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
from ._alifestd_has_contiguous_ids_polars import (
    alifestd_has_contiguous_ids_polars,
)
from ._alifestd_is_topologically_sorted_polars import (
    alifestd_is_topologically_sorted_polars,
)
from ._alifestd_mark_num_leaves_asexual import (
    _alifestd_mark_num_leaves_asexual_fast_path,
)
from ._alifestd_try_add_ancestor_id_col_polars import (
    alifestd_try_add_ancestor_id_col_polars,
)


def _alifestd_tgs_mark_clade_size_partitions_fast_path(
    *,
    clade_sizes: np.ndarray,
    neutral_gain_positions: np.ndarray,
    ratio: float,
    min_neutral_gains: int,
) -> np.ndarray:
    """Implementation detail for
    `alifestd_tgs_mark_clade_size_partitions_polars`. Acronym tgs stands for
    trait gain/selection.

    Returns each node's clade size partition, given its clade size (number
    of leaves) and the positions of nodes whose stem branches gained a
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
                "- alifestd_tgs_mark_clade_size_partitions_polars: clade "
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
            "- alifestd_tgs_mark_clade_size_partitions_polars: clade size "
            f"partition {partition} has clade sizes "
            f"{sizes.min(initial=0)}-{sizes.max(initial=0)}, {len(sizes)} "
            f"nodes, and {neutral_gains[partition]} neutral trait gains",
        )
    return partitions


def alifestd_tgs_mark_clade_size_partitions_polars(
    phylogeny_df: pl.DataFrame,
    *,
    neutral_traits: typing.Sequence[str],
    ratio: float = 2.0,
    min_neutral_gains: int = 20,
    mark_as: str = "tgs_clade_size_partition",
) -> pl.DataFrame:
    """Mark each node's clade size partition, grouping nodes by the number of
    leaves in their clade, for the `size_partitions` of
    `alifestd_tgs_allmdc_estimate_polars`.

    Clade sizes are binned on a log scale: a node whose clade has `n`
    leaves falls in bin ceil(log(n) / log(`ratio`)), so the default ratio
    of 2 gives bins of 1, 2, 3-4, 5-8, 9-16, ... leaves. Bins are numbered
    consecutively, skipping clade sizes absent from the tree, so leaves
    (clade size 1) are always partition 0.

    So that every partition has neutral trait gains to calibrate against,
    partitions are then merged from the largest clades down: the first
    partition above the leaves' with fewer than `min_neutral_gains` gains of
    `neutral_traits` on its nodes' stem branches, and every partition of
    larger clades, joins the partition below it. If even the first inner
    partition has too few gains, all inner nodes form one partition. Leaves
    always keep their own partition.

    Parameters
    ----------
    phylogeny_df : polars.DataFrame or polars.LazyFrame
        The phylogeny as a dataframe in alife standard format.

        Must represent an asexual phylogeny with contiguous,
        topologically sorted ids.
    neutral_traits : Sequence[str]
        Names of integer columns giving each neutral trait's transition on
        each node's stem branch (+1 for a gain), as for
        `alifestd_tgs_allmdc_estimate_polars`'s `trait_transitions`.
    ratio : float, default 2.0
        Ratio between successive bins' clade size bounds; must exceed 1.
        Larger ratios give fewer, larger partitions.
    min_neutral_gains : int, default 20
        Fewest neutral trait gains a partition of inner nodes may hold
        before merging; 0 disables merging.
    mark_as : str, default "tgs_clade_size_partition"
        Output column name.

    Returns
    -------
    polars.DataFrame
        The phylogeny with an added integer column of clade size
        partitions.

    Raises
    ------
    NotImplementedError
        If `phylogeny_df` has no "ancestor_id" column, or if ids are
        non-contiguous or not topologically sorted.
    ValueError
        If a `neutral_traits` column is missing, if `ratio` does not exceed
        1, or if `min_neutral_gains` is negative.

    See Also
    --------
    alifestd_tgs_allmdc_estimate_polars :
        Calibrates branch lengths within these partitions.
    """
    if not ratio > 1:
        raise ValueError(f"ratio must exceed 1, got {ratio}")
    if min_neutral_gains < 0:
        raise ValueError(
            f"min_neutral_gains must be nonnegative, got {min_neutral_gains}",
        )
    if isinstance(neutral_traits, str):
        raise TypeError(
            "neutral_traits must be a sequence of column names, not a "
            f"single string; did you mean [{neutral_traits!r}]?",
        )

    phylogeny_df = alifestd_try_add_ancestor_id_col_polars(phylogeny_df)
    schema_names = phylogeny_df.lazy().collect_schema().names()
    if "ancestor_id" not in schema_names:
        raise NotImplementedError(
            "alifestd_tgs_mark_clade_size_partitions_polars only supports "
            "asexual phylogenies.",
        )
    for column in neutral_traits:
        if column not in schema_names:
            raise ValueError(f"phylogeny_df has no column {column}")

    if phylogeny_df.lazy().limit(1).collect().is_empty():
        return phylogeny_df.with_columns(
            pl.lit(0, dtype=pl.Int64).alias(mark_as),
        )
    if not alifestd_has_contiguous_ids_polars(phylogeny_df):
        raise NotImplementedError("non-contiguous ids not supported")
    if not alifestd_is_topologically_sorted_polars(phylogeny_df):
        raise NotImplementedError(
            "non-topologically-sorted data not supported",
        )

    selected = (
        phylogeny_df.lazy().select("ancestor_id", *neutral_traits).collect()
    )
    ancestor_ids = selected["ancestor_id"].to_numpy()
    is_root = ancestor_ids == np.arange(len(ancestor_ids))
    neutral_gain_positions = np.concatenate(
        [np.zeros(0, dtype=np.int64)]
        + [
            np.flatnonzero((selected[column].to_numpy() == 1) & ~is_root)
            for column in neutral_traits
        ],
    )
    partitions = _alifestd_tgs_mark_clade_size_partitions_fast_path(
        clade_sizes=_alifestd_mark_num_leaves_asexual_fast_path(ancestor_ids),
        neutral_gain_positions=neutral_gain_positions,
        ratio=ratio,
        min_neutral_gains=min_neutral_gains,
    )
    return phylogeny_df.with_columns(
        pl.Series(name=mark_as, values=partitions, dtype=pl.Int64),
    )


_raw_description = f"""\
{os.path.basename(__file__)} | \
(phyloframe v{get_phyloframe_version()}/joinem v{joinem.__version__})

Mark each node's clade size partition, grouping nodes by the number of leaves
in their clade on a log scale, then merging partitions with too few neutral
trait gains, for `--size-partitions` of
`phyloframe.legacy._alifestd_tgs_allmdc_estimate_polars`.

Data is assumed to be in alife standard format.
Only supports asexual phylogenies.

Additional Notes
================
- Requires contiguous, topologically sorted ids.

- Use `--eager-read` if modifying data file inplace.

- This CLI entrypoint is experimental and may be subject to change.

See Also
========
phyloframe.legacy._alifestd_tgs_allmdc_estimate_polars :
    Calibrates branch lengths within these partitions.
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
            "phyloframe.legacy._alifestd_tgs_mark_clade_size_partitions_polars"
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
        "--ratio",
        default=2.0,
        type=float,
        help="Ratio between successive clade size bins (default: 2.0).",
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
        "--mark-as",
        default="tgs_clade_size_partition",
        type=str,
        help="Output column name (default: tgs_clade_size_partition).",
    )
    return parser


if __name__ == "__main__":
    begin_prod_logging()

    parser = _create_parser()
    args, __ = parser.parse_known_args()

    try:
        with log_context_duration(
            "phyloframe.legacy._alifestd_tgs_mark_clade_size_partitions_polars",
            logging.info,
        ):
            _run_dataframe_cli(
                base_parser=parser,
                output_dataframe_op=functools.partial(
                    alifestd_tgs_mark_clade_size_partitions_polars,
                    neutral_traits=args.neutral_traits,
                    ratio=args.ratio,
                    min_neutral_gains=args.min_neutral_gains,
                    mark_as=args.mark_as,
                ),
            )
    except NotImplementedError as e:
        logging.error(
            "- polars op not yet implemented, use pandas op instead",
        )
        raise e
