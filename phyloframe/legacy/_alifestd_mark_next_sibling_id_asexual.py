import argparse
import functools
import logging
import os

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
from .._auxlib._jit import jit
from .._auxlib._log_context_duration import log_context_duration
from ._alifestd_has_contiguous_ids import alifestd_has_contiguous_ids
from ._alifestd_is_topologically_sorted import alifestd_is_topologically_sorted
from ._alifestd_topological_sort import alifestd_topological_sort
from ._alifestd_try_add_ancestor_id_col import alifestd_try_add_ancestor_id_col


@jit(nopython=True)
def _alifestd_mark_next_sibling_id_asexual_fast_path(
    ancestor_ids: np.ndarray,
) -> np.ndarray:
    """Implementation detail for `alifestd_mark_next_sibling_id_asexual`.

    Returns array where each node's value is the next-highest id with the same
    parent, or own id if no such sibling exists.
    """
    n = len(ancestor_ids)
    next_sibling_ids = np.arange(n, dtype=ancestor_ids.dtype)  # default: self
    # last_child_with_parent[p] tracks the most recently seen child of p
    last_child_with_parent = np.full(n, -1, dtype=np.int64)

    for idx, parent in enumerate(ancestor_ids):
        if parent != idx:  # skip genesis/root self-ref
            prev = last_child_with_parent[parent]
            if prev != -1:
                next_sibling_ids[prev] = idx
            last_child_with_parent[parent] = idx

    return next_sibling_ids


def _alifestd_mark_next_sibling_id_asexual_slow_path(
    phylogeny_df: pd.DataFrame,
) -> pd.DataFrame:
    """Implementation detail for `alifestd_mark_next_sibling_id_asexual`."""
    phylogeny_df.index = phylogeny_df["id"]

    phylogeny_df["next_sibling_id"] = phylogeny_df["id"]

    last_child_with_parent = {}
    for idx in phylogeny_df.index:
        ancestor_id = phylogeny_df.at[idx, "ancestor_id"]
        if ancestor_id == idx:
            continue  # handle genesis cases
        if ancestor_id in last_child_with_parent:
            prev = last_child_with_parent[ancestor_id]
            phylogeny_df.at[prev, "next_sibling_id"] = idx
        last_child_with_parent[ancestor_id] = idx

    return phylogeny_df


def alifestd_mark_next_sibling_id_asexual(
    phylogeny_df: pd.DataFrame,
    mutate: bool = False,
    mark_as: str = "next_sibling_id",
) -> pd.DataFrame:
    """Add column `next_sibling_id`, the next-highest id sharing the same
    parent.

    The output column name can be changed via the ``mark_as`` parameter.

    If no such sibling exists, marks own id.

    A topological sort will be applied if `phylogeny_df` is not topologically
    sorted. Dataframe reindexing (e.g., df.index) may be applied.

    Input dataframe is not mutated by this operation unless `mutate` set True.
    If mutate set True, operation does not occur in place; still use return
    value to get transformed phylogeny dataframe.
    """

    if not mutate:
        phylogeny_df = phylogeny_df.copy()

    phylogeny_df = alifestd_try_add_ancestor_id_col(phylogeny_df, mutate=True)

    if not alifestd_is_topologically_sorted(phylogeny_df):
        phylogeny_df = alifestd_topological_sort(phylogeny_df, mutate=True)

    if alifestd_has_contiguous_ids(phylogeny_df):
        phylogeny_df[
            mark_as
        ] = _alifestd_mark_next_sibling_id_asexual_fast_path(
            phylogeny_df["ancestor_id"].to_numpy(),
        )
        return phylogeny_df
    else:
        return _alifestd_mark_next_sibling_id_asexual_slow_path(
            phylogeny_df,
        )


_raw_description = f"""{os.path.basename(__file__)} | (phyloframe v{get_phyloframe_version()}/joinem v{joinem.__version__})

Add column `next_sibling_id`, the next-highest id sharing the same parent.

Data is assumed to be in alife standard format.

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
        dfcli_module="phyloframe.legacy._alifestd_mark_next_sibling_id_asexual",
        dfcli_version=get_phyloframe_version(),
    )
    parser.add_argument(
        "--mark-as",
        default="next_sibling_id",
        type=str,
        help="output column name (default: next_sibling_id)",
    )
    return parser


if __name__ == "__main__":
    begin_prod_logging()

    parser = _create_parser()
    args, __ = parser.parse_known_args()
    with log_context_duration(
        "phyloframe.legacy._alifestd_mark_next_sibling_id_asexual",
        logging.info,
    ):
        _run_dataframe_cli(
            base_parser=parser,
            output_dataframe_op=delegate_polars_implementation()(
                functools.partial(
                    alifestd_mark_next_sibling_id_asexual, mark_as=args.mark_as
                ),
            ),
        )
