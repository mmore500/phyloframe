import argparse
import contextlib
import functools
import logging
import os
import sys
import typing
import warnings

import joinem
from joinem._dataframe_cli import _add_parser_base, _run_dataframe_cli
import numpy as np
import opytional as opyt
import pandas as pd
from tqdm import tqdm

from .._auxlib._RngStateContext import RngStateContext
from .._auxlib._add_bool_arg import add_bool_arg
from .._auxlib._begin_prod_logging import begin_prod_logging
from .._auxlib._delegate_polars_implementation import (
    delegate_polars_implementation,
)
from .._auxlib._format_cli_description import format_cli_description
from .._auxlib._get_phyloframe_version import get_phyloframe_version
from .._auxlib._log_context_duration import log_context_duration
from ._alifestd_calc_mrca_id_vector_asexual import (
    alifestd_calc_mrca_id_vector_asexual,
)
from ._alifestd_has_contiguous_ids import alifestd_has_contiguous_ids
from ._alifestd_mark_leaves import alifestd_mark_leaves
from ._alifestd_prune_extinct_lineages_asexual import (
    alifestd_prune_extinct_lineages_asexual,
)
from ._alifestd_topological_sensitivity_warned import (
    alifestd_topological_sensitivity_warned,
)
from ._alifestd_try_add_ancestor_id_col import alifestd_try_add_ancestor_id_col

_DEPRECATED_SENTINEL = "deprecated_sentinel"


def _alifestd_downsample_tips_lineage_select_target_id(
    is_leaf: np.ndarray,
    target_values: np.ndarray,
) -> int:
    """Select the target leaf id (largest target value, ties broken by RNG).

    Uses numpy's global RNG for tie-breaking.  Callers should wrap with
    ``RngStateContext`` when deterministic behaviour is required.

    Parameters
    ----------
    is_leaf : numpy.ndarray
        Boolean array indicating which taxa are leaves.
    target_values : numpy.ndarray
        Values used to select the target leaf (all taxa).

    Returns
    -------
    int
        The id of the selected target leaf.
    """
    max_target = target_values[is_leaf].max()
    logging.info(
        "_alifestd_downsample_tips_lineage_select_target_id: "
        f"max target criterion {max_target=}",
    )
    candidate_ids = np.flatnonzero(is_leaf & (target_values == max_target))
    logging.info(
        "_alifestd_downsample_tips_lineage_select_target_id: "
        f"selecting from {len(candidate_ids)=}",
    )
    return int(np.random.choice(candidate_ids))


def _alifestd_downsample_tips_lineage_impl(
    is_leaf: np.ndarray,
    criterion_values: np.ndarray,
    n_downsample: int,
    mrca_vector: np.ndarray,
) -> np.ndarray:
    """Shared numpy implementation for lineage-based tip downsampling.

    Computes off-lineage deltas from a pre-computed MRCA vector and
    returns a boolean extant mask.

    Parameters
    ----------
    is_leaf : numpy.ndarray
        Boolean array indicating which taxa are leaves.
    criterion_values : numpy.ndarray
        Values used to compute off-lineage delta (all taxa).
    n_downsample : int
        Number of tips to retain.
    mrca_vector : numpy.ndarray
        Integer array of MRCA ids for each taxon with respect to the
        target leaf.  Taxa in a different tree should have ``-1``.

    Returns
    -------
    numpy.ndarray
        Boolean array of length ``len(is_leaf)`` marking retained taxa.
    """
    # Taxa with no common ancestor (different tree) get -1 from MRCA
    # calc; replace with a safe dummy id (0) so the lookup doesn't fail,
    # then exclude these taxa from selection below.
    no_mrca_mask = mrca_vector == -1
    safe_mrca = np.where(no_mrca_mask, 0, mrca_vector)

    logging.info(
        "_alifestd_downsample_tips_lineage_impl: "
        "calculating off lineage delta...",
    )
    off_lineage_delta = np.abs(
        criterion_values - criterion_values[safe_mrca],
    )

    # Select eligible leaves with the smallest deltas
    logging.info(
        "_alifestd_downsample_tips_lineage_impl: "
        "filtering leaf eligibility...",
    )
    is_eligible = is_leaf & ~no_mrca_mask
    eligible_ids = np.flatnonzero(is_eligible)
    eligible_deltas = off_lineage_delta[is_eligible]

    logging.info(
        "_alifestd_downsample_tips_lineage_impl: selecting kept ids...",
    )
    if n_downsample >= len(eligible_deltas):
        kept_ids = eligible_ids
    else:
        partition_idx = np.argpartition(eligible_deltas, n_downsample)[
            :n_downsample
        ]
        kept_ids = eligible_ids[partition_idx]

    logging.info(
        "_alifestd_downsample_tips_lineage_impl: building extant mask...",
    )
    return np.bincount(kept_ids, minlength=len(is_leaf)).astype(bool)


@alifestd_topological_sensitivity_warned(
    insert=False,
    delete=True,
    update=False,
)
def alifestd_downsample_tips_lineage_asexual(
    phylogeny_df: pd.DataFrame,
    n_downsample: typing.Any = _DEPRECATED_SENTINEL,
    mutate: bool = False,
    seed: typing.Optional[int] = None,
    *,
    num_tips: typing.Any = _DEPRECATED_SENTINEL,
    criterion_delta: str = "origin_time",
    criterion_target: str = "origin_time",
    progress_wrap: typing.Callable = lambda x: x,
) -> pd.DataFrame:
    """Retain the `n_downsample` leaves closest to the lineage of a target
    leaf.

    Selects a target leaf as the leaf with the largest `criterion_target`
    value (ties broken randomly). For each leaf, the most recent common
    ancestor (MRCA) with the target leaf is identified and the "off-lineage
    delta" is computed as the absolute difference between the leaf's
    `criterion_delta` value and its MRCA's `criterion_delta` value. The
    `n_downsample` leaves with the smallest off-lineage deltas are retained.

    If `n_downsample` is greater than or equal to the number of leaves in
    the phylogeny, the whole phylogeny is returned. Ties in off-lineage
    delta are broken arbitrarily.

    Only supports asexual phylogenies.

    Parameters
    ----------
    phylogeny_df : pandas.DataFrame
        The phylogeny as a dataframe in alife standard format.

        Must represent an asexual phylogeny.
    n_downsample : int
        Number of tips to retain.
    mutate : bool, default False
        Are side effects on the input argument `phylogeny_df` allowed?
    seed : int, optional
        Random seed for reproducible target-leaf selection when there are
        ties in `criterion_target`.
    criterion_delta : str, default "origin_time"
        Column name used to compute the off-lineage delta for each leaf.
        The delta is the absolute difference between a leaf's value and
        its MRCA's value in this column.
    criterion_target : str, default "origin_time"
        Column name used to select the target leaf. The leaf with the
        largest value in this column is chosen as the target. Note that
        ties are broken by random sample, allowing a seed to be
        provided.
    progress_wrap : Callable, optional
        Pass tqdm or equivalent to display a progress bar.

    Raises
    ------
    ValueError
        If `criterion_delta` or `criterion_target` is not a column in
        `phylogeny_df`.

    Returns
    -------
    pandas.DataFrame
        The pruned phylogeny in alife standard format.
    """
    if num_tips != _DEPRECATED_SENTINEL:
        warnings.warn(
            "num_tips is deprecated in favor of n_downsample and "
            "will be removed in a future release of phyloframe.",
            DeprecationWarning,
            stacklevel=2,
        )
        if n_downsample != _DEPRECATED_SENTINEL:
            raise TypeError(
                "cannot specify both n_downsample and num_tips",
            )
        n_downsample = num_tips
    elif n_downsample == _DEPRECATED_SENTINEL:
        raise TypeError(
            "alifestd_downsample_tips_lineage_asexual() missing required "
            "argument: 'n_downsample'",
        )

    for criterion in (criterion_delta, criterion_target):
        if criterion not in phylogeny_df.columns:
            raise ValueError(
                f"criterion column {criterion!r} not found in phylogeny_df",
            )

    if not mutate:
        phylogeny_df = phylogeny_df.copy()

    if phylogeny_df.empty:
        return phylogeny_df

    logging.info(
        "- alifestd_downsample_tips_lineage_asexual: "
        "adding ancestor_id col...",
    )
    phylogeny_df = alifestd_try_add_ancestor_id_col(phylogeny_df)
    if "ancestor_id" not in phylogeny_df.columns:
        raise ValueError(
            "alifestd_downsample_tips_lineage_asexual only supports "
            "asexual phylogenies.",
        )

    logging.info(
        "- alifestd_downsample_tips_lineage_asexual: marking leaves...",
    )
    if "is_leaf" not in phylogeny_df.columns:
        phylogeny_df = alifestd_mark_leaves(phylogeny_df)

    logging.info(
        "- alifestd_downsample_tips_lineage_asexual: "
        "checking contiguous ids...",
    )
    if alifestd_has_contiguous_ids(phylogeny_df):
        phylogeny_df.reset_index(drop=True, inplace=True)
    else:
        # non-contiguous id branch not tested because
        # alifestd_calc_mrca_id_vector_asexual doesn't support it
        raise NotImplementedError(
            "non-contiguous ids not yet supported",
        )

    is_leaf = phylogeny_df["is_leaf"].to_numpy()
    target_values = phylogeny_df[criterion_target].to_numpy()
    criterion_values = phylogeny_df[criterion_delta].to_numpy()

    with opyt.apply_if_or_else(seed, RngStateContext, contextlib.nullcontext):
        target_id = _alifestd_downsample_tips_lineage_select_target_id(
            is_leaf, target_values
        )

    logging.info(
        "- alifestd_downsample_tips_lineage_asexual: "
        "computing mrca vector...",
    )
    mrca_vector = alifestd_calc_mrca_id_vector_asexual(
        phylogeny_df, target_id=target_id, progress_wrap=progress_wrap
    )

    logging.info(
        "- alifestd_downsample_tips_lineage_asexual: "
        "computing lineage downsample...",
    )
    is_extant = _alifestd_downsample_tips_lineage_impl(
        is_leaf=is_leaf,
        criterion_values=criterion_values,
        n_downsample=n_downsample,
        mrca_vector=mrca_vector,
    )

    logging.info(
        "- alifestd_downsample_tips_lineage_asexual: pruning...",
    )
    phylogeny_df["extant"] = is_extant
    return alifestd_prune_extinct_lineages_asexual(
        phylogeny_df, mutate=True
    ).drop(columns=["extant"])


_raw_description = f"""{os.path.basename(__file__)} | (phyloframe v{get_phyloframe_version()}/joinem v{joinem.__version__})

Retain the `-n` leaves closest to the lineage of a target leaf.

The target leaf is chosen as the leaf with the largest
`--criterion-target` value. For each leaf, the off-lineage delta is
the absolute difference between the leaf's `--criterion-delta` value
and its MRCA's `--criterion-delta` value with respect to the target.
The `-n` leaves with the smallest deltas are retained.

If `-n` is greater than or equal to the number of leaves in the phylogeny, the whole phylogeny is returned. Ties are broken arbitrarily.

Data is assumed to be in alife standard format.
Only supports asexual phylogenies.

Additional Notes
================
- Requires 'ancestor_id' column to be present in input DataFrame.
Otherwise, no action is taken.

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
        dfcli_module="phyloframe.legacy._alifestd_downsample_tips_lineage_asexual",
        dfcli_version=get_phyloframe_version(),
    )
    parser.add_argument(
        "-n",
        default=sys.maxsize,
        type=int,
        help="Number of tips to retain.",
    )
    parser.add_argument(
        "--criterion-delta",
        default="origin_time",
        type=str,
        help="Column used to compute off-lineage delta (default: origin_time).",
    )
    parser.add_argument(
        "--criterion-target",
        default="origin_time",
        type=str,
        help="Column used to select the target leaf (default: origin_time).",
    )
    parser.add_argument(
        "--seed",
        default=None,
        dest="seed",
        help="Integer seed for deterministic target-leaf selection.",
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
    with log_context_duration(
        "phyloframe.legacy._alifestd_downsample_tips_lineage_asexual",
        logging.info,
    ):
        _run_dataframe_cli(
            base_parser=parser,
            output_dataframe_op=delegate_polars_implementation()(
                functools.partial(
                    alifestd_downsample_tips_lineage_asexual,
                    n_downsample=args.n,
                    seed=args.seed,
                    criterion_delta=args.criterion_delta,
                    criterion_target=args.criterion_target,
                    progress_wrap=tqdm,
                    ignore_topological_sensitivity=args.ignore_topological_sensitivity,
                    drop_topological_sensitivity=args.drop_topological_sensitivity,
                ),
            ),
            overridden_arguments="ignore",  # seed is overridden
        )
