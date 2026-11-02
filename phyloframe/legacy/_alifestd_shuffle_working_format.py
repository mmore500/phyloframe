import argparse
import functools
import logging
import os
import typing

import joinem
from joinem._dataframe_cli import _add_parser_base, _run_dataframe_cli
import numpy as np
import pandas as pd

from .._auxlib._begin_prod_logging import begin_prod_logging
from .._auxlib._delegate_polars_implementation import (
    delegate_polars_implementation,
)
from .._auxlib._format_cli_description import format_cli_description
from .._auxlib._get_phyloframe_version import get_phyloframe_version
from .._auxlib._log_context_duration import log_context_duration
from ._alifestd_assign_contiguous_ids import alifestd_assign_contiguous_ids
from ._alifestd_topological_sort import alifestd_topological_sort
from ._alifestd_try_add_ancestor_id_col import alifestd_try_add_ancestor_id_col


def alifestd_shuffle_working_format(
    phylogeny_df: pd.DataFrame,
    mutate: bool = False,
    *,
    seed: typing.Optional[int] = None,
) -> pd.DataFrame:
    """Randomly reorder rows, then restore working format.

    The returned phylogeny dataframe represents the same phylogeny, but with
    rows in a random topologically sorted order and ids reassigned to be
    contiguous (i.e., organisms' ids correspond to row number). Useful for
    testing that operations do not depend on row order or id assignment.

    Supports sexual and asexual phylogenies. Track taxa across the shuffle
    with a label column, as `id` values are reassigned.

    Input dataframe is not mutated by this operation unless `mutate` set True.
    If mutate set True, operation does not occur in place; still use return
    value to get transformed phylogeny dataframe.

    See Also
    --------
    alifestd_shuffle_working_format_polars :
        Polars-based implementation.
    alifestd_to_working_format :
        Restores working format without shuffling.
    """
    if not mutate:
        phylogeny_df = phylogeny_df.copy()

    phylogeny_df = alifestd_try_add_ancestor_id_col(phylogeny_df, mutate=True)

    permutation = np.random.default_rng(seed).permutation(len(phylogeny_df))
    phylogeny_df = phylogeny_df.iloc[permutation].reset_index(drop=True)

    # relabel ids to shuffled row positions, so that topological sort, which
    # orders by id, inherits shuffled order...
    phylogeny_df = alifestd_assign_contiguous_ids(phylogeny_df, mutate=True)
    phylogeny_df = alifestd_topological_sort(phylogeny_df, mutate=True)
    # ... then relabel ids to sorted row positions
    phylogeny_df = alifestd_assign_contiguous_ids(phylogeny_df, mutate=True)

    return phylogeny_df.reset_index(drop=True)


_raw_description = f"""{os.path.basename(__file__)} | (phyloframe v{get_phyloframe_version()}/joinem v{joinem.__version__})

Randomly reorder rows, then restore working format (topologically sorted, contiguous ids).

Data is assumed to be in alife standard format.

Additional Notes
================
- Use `--seed` for reproducible shuffling.

- Use `--eager-read` if modifying data file inplace.

- This CLI entrypoint is experimental and may be subject to change.

See Also
========
phyloframe.legacy._alifestd_shuffle_working_format_polars :
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
        dfcli_module="phyloframe.legacy._alifestd_shuffle_working_format",
        dfcli_version=get_phyloframe_version(),
    )
    parser.add_argument(
        "--seed",
        default=None,
        type=int,
        help="random seed (default: None)",
    )
    return parser


if __name__ == "__main__":
    begin_prod_logging()

    parser = _create_parser()
    args, __ = parser.parse_known_args()
    with log_context_duration(
        "phyloframe.legacy._alifestd_shuffle_working_format", logging.info
    ):
        _run_dataframe_cli(
            base_parser=parser,
            output_dataframe_op=delegate_polars_implementation()(
                functools.partial(
                    alifestd_shuffle_working_format,
                    seed=args.seed,
                ),
            ),
        )
