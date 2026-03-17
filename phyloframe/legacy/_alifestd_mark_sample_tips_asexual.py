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
from ._alifestd_find_leaf_ids import alifestd_find_leaf_ids
from ._alifestd_has_contiguous_ids import alifestd_has_contiguous_ids
from ._alifestd_try_add_ancestor_id_col import alifestd_try_add_ancestor_id_col


def _alifestd_mark_sample_tips_asexual_impl(
    phylogeny_df: pd.DataFrame,
    n_sample: int,
    mark_as: str,
) -> pd.DataFrame:
    """Implementation detail for alifestd_mark_sample_tips_asexual."""
    tips = alifestd_find_leaf_ids(phylogeny_df)
    kept = np.random.choice(tips, min(n_sample, len(tips)), replace=False)
    if alifestd_has_contiguous_ids(phylogeny_df):
        extant = np.zeros(len(phylogeny_df), dtype=bool)
        extant[kept] = True
        phylogeny_df[mark_as] = extant
    else:
        phylogeny_df[mark_as] = phylogeny_df["id"].isin(kept)

    return phylogeny_df


def alifestd_mark_sample_tips_asexual(
    phylogeny_df: pd.DataFrame,
    n_sample: int,
    mutate: bool = False,
    seed: typing.Optional[int] = None,
    *,
    mark_as: str = "alifestd_mark_sample_tips_asexual",
) -> pd.DataFrame:
    """Mark a random subsample of `n_sample` tips.

    Adds a boolean column ``mark_as`` indicating retained tips.

    If `n_sample` is greater than the number of tips in the phylogeny,
    all tips are marked.

    Only supports asexual phylogenies.
    """
    if not mutate:
        phylogeny_df = phylogeny_df.copy()

    phylogeny_df = alifestd_try_add_ancestor_id_col(phylogeny_df)
    if "ancestor_id" not in phylogeny_df.columns:
        raise ValueError(
            "alifestd_mark_sample_tips_asexual only supports "
            "asexual phylogenies.",
        )

    if phylogeny_df.empty:
        phylogeny_df[mark_as] = pd.Series(dtype=bool)
        return phylogeny_df

    impl = (
        with_rng_state_context(seed)(_alifestd_mark_sample_tips_asexual_impl)
        if seed is not None
        else _alifestd_mark_sample_tips_asexual_impl
    )

    return impl(phylogeny_df, n_sample, mark_as)


_raw_description = f"""{os.path.basename(__file__)} | (phyloframe v{get_phyloframe_version()}/joinem v{joinem.__version__})

Mark a random subsample of `-n` tips with a boolean column.

If `-n` is greater than the number of tips in the phylogeny, all tips are marked.

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
phyloframe.legacy._alifestd_mark_sample_tips_polars :
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
        dfcli_module="phyloframe.legacy._alifestd_mark_sample_tips_asexual",
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
        default="alifestd_mark_sample_tips_asexual",
        type=str,
        help="Column name for the boolean mark (default: alifestd_mark_sample_tips_asexual).",
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
        "phyloframe.legacy._alifestd_mark_sample_tips_asexual", logging.info
    ):
        _run_dataframe_cli(
            base_parser=parser,
            output_dataframe_op=delegate_polars_implementation()(
                functools.partial(
                    alifestd_mark_sample_tips_asexual,
                    n_sample=args.n,
                    seed=args.seed,
                    mark_as=args.mark_as,
                ),
            ),
            overridden_arguments="ignore",  # seed is overridden
        )
