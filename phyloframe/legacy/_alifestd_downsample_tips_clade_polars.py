import argparse
import contextlib
import functools
import gc
import logging
import os
import sys
import typing

import joinem
from joinem._dataframe_cli import _add_parser_base, _run_dataframe_cli
import numpy as np
import opytional as opyt
import polars as pl

from .._auxlib._RngStateContext import RngStateContext
from .._auxlib._add_bool_arg import add_bool_arg
from .._auxlib._begin_prod_logging import begin_prod_logging
from .._auxlib._format_cli_description import format_cli_description
from .._auxlib._get_phyloframe_version import get_phyloframe_version
from .._auxlib._log_context_duration import log_context_duration
from .._auxlib._log_memory_usage import log_memory_usage
from ._alifestd_has_contiguous_ids_polars import (
    alifestd_has_contiguous_ids_polars,
)
from ._alifestd_is_topologically_sorted_polars import (
    alifestd_is_topologically_sorted_polars,
)
from ._alifestd_mark_leaves_polars import alifestd_mark_leaves_polars
from ._alifestd_mark_num_leaves_polars import (
    alifestd_mark_num_leaves_polars,
)
from ._alifestd_mask_descendants_polars import (
    alifestd_mask_descendants_polars,
)
from ._alifestd_prune_extinct_lineages_polars import (
    alifestd_prune_extinct_lineages_polars,
)
from ._alifestd_topological_sensitivity_warned_polars import (
    alifestd_topological_sensitivity_warned_polars,
)


def _alifestd_downsample_tips_clade_polars_impl(
    phylogeny_df: pl.DataFrame,
    n_downsample: int,
) -> pl.DataFrame:
    """Implementation detail for alifestd_downsample_tips_clade_polars."""

    logging.info(
        "- alifestd_downsample_tips_clade_polars: marking leaves...",
    )
    phylogeny_df = alifestd_mark_leaves_polars(phylogeny_df)
    gc.collect()
    log_memory_usage(logging.info)

    logging.info(
        "- alifestd_downsample_tips_clade_polars: marking num_leaves...",
    )
    phylogeny_df = alifestd_mark_num_leaves_polars(phylogeny_df)
    gc.collect()
    log_memory_usage(logging.info)

    logging.info(
        "- alifestd_downsample_tips_clade_polars: "
        "collecting ancestor_id values...",
    )
    ancestor_ids = (
        phylogeny_df.lazy()
        .select("ancestor_id")
        .collect()
        .to_series()
        .to_numpy()
    )
    num_leaves = (
        phylogeny_df.lazy()
        .select("num_leaves")
        .collect()
        .to_series()
        .to_numpy()
    )
    gc.collect()
    log_memory_usage(logging.info)

    logging.info(
        "- alifestd_downsample_tips_clade_polars: finding candidates...",
    )
    is_candidate = num_leaves <= n_downsample
    if is_candidate.all():
        return phylogeny_df

    is_candidate &= num_leaves[ancestor_ids] > n_downsample
    gc.collect()
    log_memory_usage(logging.info)

    logging.info(
        "- alifestd_downsample_tips_clade_polars: "
        "sampling weighted candidate...",
    )
    ids = phylogeny_df.lazy().select("id").collect().to_series().to_numpy()
    candidate_ids = ids[is_candidate]
    candidate_num_leaves = num_leaves[is_candidate]
    cumulative_weights = np.cumsum(candidate_num_leaves)
    total_weight = cumulative_weights[-1]
    sampled_idx = np.searchsorted(cumulative_weights, np.random.randint(total_weight), side="right")
    sampled = candidate_ids[sampled_idx]
    del ids, candidate_ids, candidate_num_leaves, cumulative_weights
    gc.collect()
    log_memory_usage(logging.info)

    del ancestor_ids, num_leaves
    gc.collect()
    log_memory_usage(logging.info)

    logging.info(
        "- alifestd_downsample_tips_clade_polars: "
        "marking descendants of sampled clade...",
    )
    n_rows = phylogeny_df.lazy().select(pl.len()).collect().item()
    ancestor_mask = np.zeros(n_rows, dtype=bool)
    ancestor_mask[sampled] = True
    phylogeny_df = alifestd_mask_descendants_polars(
        phylogeny_df,
        ancestor_mask=ancestor_mask,
    )
    del ancestor_mask
    gc.collect()
    log_memory_usage(logging.info)

    logging.info(
        "- alifestd_downsample_tips_clade_polars: marking extant...",
    )
    is_leaf = (
        phylogeny_df.lazy().select("is_leaf").collect().to_series().to_numpy()
    )
    is_descendant = (
        phylogeny_df.lazy()
        .select("alifestd_mask_descendants_polars")
        .collect()
        .to_series()
        .to_numpy()
    )
    extant = is_descendant & is_leaf
    phylogeny_df = phylogeny_df.with_columns(extant=extant).drop(
        "alifestd_mask_descendants_polars",
    )
    del is_descendant, is_leaf, extant
    gc.collect()
    log_memory_usage(logging.info)

    logging.info(
        "- alifestd_downsample_tips_clade_polars: pruning...",
    )
    return alifestd_prune_extinct_lineages_polars(phylogeny_df).drop(
        "extant",
    )


@alifestd_topological_sensitivity_warned_polars(
    insert=False,
    delete=True,
    update=False,
)
def alifestd_downsample_tips_clade_polars(
    phylogeny_df: pl.DataFrame,
    n_downsample: int,
    seed: typing.Optional[int] = None,
) -> pl.DataFrame:
    """Create a subsample phylogeny containing at most `n_downsample` tips,
    comprising a single clade within the original phylogeny. Candidate clades
    are sampled proportionally to their size.

    If `n_downsample` is greater than the number of tips in the phylogeny,
    the whole phylogeny is returned.

    Only supports asexual phylogenies.

    Parameters
    ----------
    phylogeny_df : polars.DataFrame
        The phylogeny as a dataframe in alife standard format.

        Must represent an asexual phylogeny.
    n_downsample : int
        Number of tips to retain.
    seed : int, optional
        Integer seed for deterministic behavior.

    Raises
    ------
    NotImplementedError
        If `phylogeny_df` has no "ancestor_id" column or if ids are
        non-contiguous or not topologically sorted.

    Returns
    -------
    polars.DataFrame
        The downsampled phylogeny in alife standard format.

    See Also
    --------
    alifestd_downsample_tips_clade_asexual :
        Pandas-based implementation.
    """
    if "ancestor_id" not in phylogeny_df.lazy().collect_schema().names():
        raise NotImplementedError("ancestor_id column required")

    if phylogeny_df.lazy().limit(1).collect().is_empty():
        return phylogeny_df

    logging.info(
        "- alifestd_downsample_tips_clade_polars: "
        "checking contiguous ids...",
    )
    if not alifestd_has_contiguous_ids_polars(phylogeny_df):
        raise NotImplementedError(
            "non-contiguous ids not yet supported",
        )

    logging.info(
        "- alifestd_downsample_tips_clade_polars: "
        "checking topological sort...",
    )
    if not alifestd_is_topologically_sorted_polars(phylogeny_df):
        raise NotImplementedError(
            "topologically unsorted rows not yet supported",
        )

    with opyt.apply_if_or_else(seed, RngStateContext, contextlib.nullcontext):
        return _alifestd_downsample_tips_clade_polars_impl(
            phylogeny_df,
            n_downsample,
        )


_raw_description = f"""{os.path.basename(__file__)} | (phyloframe v{get_phyloframe_version()}/joinem v{joinem.__version__})

Create a subsample phylogeny containing at most `-n` tips, comprising a single clade within the original phylogeny.
Candidate clades are sampled proportionally to their tip count.

If `-n` is greater than the number of tips in the phylogeny, the whole phylogeny is returned.

Data is assumed to be in alife standard format.
Only supports asexual phylogenies.

Additional Notes
================
- Requires 'ancestor_id' column to be present in input DataFrame.
Otherwise, no action is taken.

- Use `--eager-read` if modifying data file inplace.

- This CLI entrypoint is experimental and may be subject to change.

See Also
========
phyloframe.legacy._alifestd_downsample_tips_clade_asexual :
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
        dfcli_module="phyloframe.legacy._alifestd_downsample_tips_clade_polars",
        dfcli_version=get_phyloframe_version(),
    )
    parser.add_argument(
        "-n",
        default=sys.maxsize,
        type=int,
        help="Number of tips to subsample.",
    )
    parser.add_argument(
        "--seed",
        default=None,
        dest="seed",
        help="Integer seed for deterministic behavior.",
        type=int,
    )
    add_bool_arg(
        parser,
        "ignore-topological-sensitivity",
        default=False,
        help="suppress topological sensitivity warning (default: False)",
    )
    add_bool_arg(
        parser,
        "drop-topological-sensitivity",
        default=False,
        help="drop topology-sensitive columns from output (default: False)",
    )
    return parser


if __name__ == "__main__":
    begin_prod_logging()

    parser = _create_parser()
    args, __ = parser.parse_known_args()

    try:
        with log_context_duration(
            "phyloframe.legacy._alifestd_downsample_tips_clade_polars",
            logging.info,
        ):
            _run_dataframe_cli(
                base_parser=parser,
                output_dataframe_op=functools.partial(
                    alifestd_downsample_tips_clade_polars,
                    n_downsample=args.n,
                    seed=args.seed,
                    ignore_topological_sensitivity=args.ignore_topological_sensitivity,
                    drop_topological_sensitivity=args.drop_topological_sensitivity,
                ),
                overridden_arguments="ignore",  # seed is overridden
            )
    except NotImplementedError as e:
        logging.error(
            "- polars op not yet implemented, use pandas op CLI instead",
        )
        raise e
