import argparse
import functools
import logging
import os
import sys
import typing

import joinem
from joinem._dataframe_cli import _add_parser_base, _run_dataframe_cli
import numpy as np
import pandas as pd

from .._auxlib._add_bool_arg import add_bool_arg
from .._auxlib._begin_prod_logging import begin_prod_logging
from .._auxlib._delegate_polars_implementation import (
    delegate_polars_implementation,
)
from .._auxlib._format_cli_description import format_cli_description
from .._auxlib._get_phyloframe_version import get_phyloframe_version
from .._auxlib._log_context_duration import log_context_duration
from .._auxlib._with_rng_state_context import with_rng_state_context
from ._alifestd_has_contiguous_ids import alifestd_has_contiguous_ids
from ._alifestd_mark_leaves import alifestd_mark_leaves
from ._alifestd_mark_num_leaves_asexual import alifestd_mark_num_leaves_asexual
from ._alifestd_mask_descendants_asexual import (
    alifestd_mask_descendants_asexual,
)
from ._alifestd_try_add_ancestor_id_col import alifestd_try_add_ancestor_id_col


def _alifestd_mark_sample_tips_clade_asexual_impl(
    phylogeny_df: pd.DataFrame,
    n_sample: int,
    mark_as: str,
) -> pd.DataFrame:
    """Implementation detail for alifestd_mark_sample_tips_clade_asexual."""
    if "is_leaf" not in phylogeny_df.columns:
        phylogeny_df = alifestd_mark_leaves(phylogeny_df, mutate=True)

    if "num_leaves" not in phylogeny_df.columns:
        phylogeny_df = alifestd_mark_num_leaves_asexual(
            phylogeny_df, mutate=True
        )

    is_candidate = phylogeny_df["num_leaves"].values <= n_sample
    if is_candidate.all():
        phylogeny_df[mark_as] = phylogeny_df["is_leaf"]
        return phylogeny_df

    if alifestd_has_contiguous_ids(phylogeny_df):
        is_candidate &= (
            phylogeny_df["num_leaves"].values[
                phylogeny_df["ancestor_id"].values.astype(np.intp)
            ]
            > n_sample
        )
    else:
        phylogeny_df.set_index("id", drop=False, inplace=True)
        is_candidate &= (
            phylogeny_df.loc[phylogeny_df["ancestor_id"], "num_leaves"].values
            > n_sample
        )

    candidate_ids = phylogeny_df.loc[is_candidate, "id"].values
    candidate_num_leaves = phylogeny_df.loc[is_candidate, "num_leaves"].values
    cumulative_weights = np.cumsum(candidate_num_leaves)
    total_weight = cumulative_weights[-1]
    sampled_idx = np.searchsorted(
        cumulative_weights, np.random.randint(total_weight), side="right"
    )
    sampled = candidate_ids[sampled_idx]

    phylogeny_df = alifestd_mask_descendants_asexual(
        phylogeny_df,
        mutate=True,
        ancestor_mask=phylogeny_df["id"].values == sampled,
    )
    phylogeny_df[mark_as] = phylogeny_df["alifestd_mask_descendants_asexual"]

    return phylogeny_df.drop(columns=["alifestd_mask_descendants_asexual"])


def alifestd_mark_sample_tips_clade_asexual(
    phylogeny_df: pd.DataFrame,
    n_sample: int,
    mutate: bool = False,
    seed: typing.Optional[int] = None,
    mark_as: str = "alifestd_mark_sample_tips_clade_asexual",
) -> pd.DataFrame:
    """Mark tips belonging to a randomly sampled clade of at most
    `n_sample` tips.

    Adds a boolean column ``mark_as`` indicating retained tips.
    Candidate clades are sampled proportionally to their size.

    If `n_sample` is greater than the number of tips in the phylogeny,
    all tips are marked.

    Only supports asexual phylogenies.
    """
    if not mutate:
        phylogeny_df = phylogeny_df.copy()

    phylogeny_df = alifestd_try_add_ancestor_id_col(phylogeny_df)
    if "ancestor_id" not in phylogeny_df.columns:
        raise ValueError(
            "alifestd_mark_sample_tips_clade_asexual only supports "
            "asexual phylogenies.",
        )

    if phylogeny_df.empty:
        phylogeny_df[mark_as] = pd.Series(dtype=bool)
        return phylogeny_df

    impl = (
        with_rng_state_context(seed)(
            _alifestd_mark_sample_tips_clade_asexual_impl
        )
        if seed is not None
        else _alifestd_mark_sample_tips_clade_asexual_impl
    )

    return impl(phylogeny_df, n_sample, mark_as)


_raw_description = f"""{os.path.basename(__file__)} | (phyloframe v{get_phyloframe_version()}/joinem v{joinem.__version__})

Mark tips belonging to a randomly sampled clade of at most `-n` tips.

Candidate clades are sampled proportionally to their tip count.

If `-n` is greater than the number of tips in the phylogeny, all tips are marked.

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
        dfcli_module="phyloframe.legacy._alifestd_mark_sample_tips_clade_asexual",
        dfcli_version=get_phyloframe_version(),
    )
    parser.add_argument(
        "-n",
        default=sys.maxsize,
        type=int,
        help="Number of tips to mark.",
    )
    parser.add_argument(
        "--seed",
        default=None,
        dest="seed",
        help="Integer seed for deterministic behavior.",
        type=int,
    )
    parser.add_argument(
        "--mark-as",
        default="alifestd_mark_sample_tips_clade_asexual",
        type=str,
        help="Column name for the boolean mark (default: alifestd_mark_sample_tips_clade_asexual).",
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
        "phyloframe.legacy._alifestd_mark_sample_tips_clade_asexual",
        logging.info,
    ):
        _run_dataframe_cli(
            base_parser=parser,
            output_dataframe_op=delegate_polars_implementation()(
                functools.partial(
                    alifestd_mark_sample_tips_clade_asexual,
                    n_sample=args.n,
                    seed=args.seed,
                    mark_as=args.mark_as,
                ),
            ),
            overridden_arguments="ignore",  # seed is overridden
        )
