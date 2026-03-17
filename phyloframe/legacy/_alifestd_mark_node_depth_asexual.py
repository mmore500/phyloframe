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
def _alifestd_calc_node_depth_asexual_contiguous(
    ancestor_ids: np.ndarray,
) -> np.ndarray:
    """Optimized implementation for asexual phylogenies with contiguous ids."""
    ancestor_ids = ancestor_ids.astype(np.uint64)
    node_depths = np.full_like(ancestor_ids, -1, dtype=np.int64)

    for id_, _ in enumerate(ancestor_ids):
        ancestor_id = ancestor_ids[id_]
        ancestor_depth = node_depths[ancestor_id]
        node_depths[id_] = ancestor_depth + 1

    return node_depths


def alifestd_mark_node_depth_asexual(
    phylogeny_df: pd.DataFrame,
    mutate: bool = False,
    *,
    mark_as: str = "node_depth",
) -> pd.DataFrame:
    """Add column `node_depth`, counting the number of nodes between a node
    and the root.

    The output column name can be changed via the ``mark_as`` parameter.

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
        # optimized implementation for contiguous ids
        node_depths = _alifestd_calc_node_depth_asexual_contiguous(
            phylogeny_df["ancestor_id"].values,
        )
        phylogeny_df[mark_as] = node_depths
        return phylogeny_df

    # slower fallback implementation
    phylogeny_df.index = phylogeny_df["id"]

    phylogeny_df[mark_as] = -1

    for idx in phylogeny_df.index:
        ancestor_id = phylogeny_df.at[idx, "ancestor_id"]
        ancestor_depth = phylogeny_df.at[ancestor_id, mark_as]
        phylogeny_df.at[idx, mark_as] = ancestor_depth + 1

    return phylogeny_df


_raw_description = f"""{os.path.basename(__file__)} | (phyloframe v{get_phyloframe_version()}/joinem v{joinem.__version__})

Add column `node_depth`, counting the number of nodes between a node and the root.

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
        dfcli_module="phyloframe.legacy._alifestd_mark_node_depth_asexual",
        dfcli_version=get_phyloframe_version(),
    )
    parser.add_argument(
        "--mark-as",
        default="node_depth",
        type=str,
        help="output column name (default: node_depth)",
    )
    return parser


if __name__ == "__main__":
    begin_prod_logging()

    parser = _create_parser()
    args, __ = parser.parse_known_args()
    with log_context_duration(
        "phyloframe.legacy._alifestd_mark_node_depth_asexual", logging.info
    ):
        _run_dataframe_cli(
            base_parser=parser,
            output_dataframe_op=delegate_polars_implementation()(
                functools.partial(
                    alifestd_mark_node_depth_asexual, mark_as=args.mark_as
                ),
            ),
        )
