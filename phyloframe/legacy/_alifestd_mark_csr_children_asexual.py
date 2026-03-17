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
from ._alifestd_mark_csr_offsets_asexual import (
    _alifestd_mark_csr_offsets_asexual_fast_path,
)
from ._alifestd_topological_sort import alifestd_topological_sort
from ._alifestd_try_add_ancestor_id_col import alifestd_try_add_ancestor_id_col


@jit(nopython=True)
def _alifestd_mark_csr_children_asexual_fast_path(
    ancestor_ids: np.ndarray,
    csr_offsets: np.ndarray,
) -> np.ndarray:
    """Implementation detail for `alifestd_mark_csr_children_asexual`.

    Returns flat array of child ids grouped by parent, ordered according
    to CSR offsets given by `csr_offsets`. Length n; entries beyond the
    total number of non-root nodes are unused.
    """
    n = len(ancestor_ids)
    csr_children = np.full(n, -1, dtype=np.int64)
    insert_pos = csr_offsets.copy()
    for i in range(n):
        p = ancestor_ids[i]
        if p != i:
            csr_children[insert_pos[p]] = i
            insert_pos[p] += 1

    return csr_children


def _alifestd_mark_csr_children_asexual_slow_path(
    phylogeny_df: pd.DataFrame,
    mark_as: str = "csr_children",
) -> pd.DataFrame:
    """Implementation detail for `alifestd_mark_csr_children_asexual`."""
    phylogeny_df.index = phylogeny_df["id"]

    # Count children per node to build offsets
    child_counts = {}
    for idx in phylogeny_df.index:
        child_counts[idx] = 0
    for idx in phylogeny_df.index:
        ancestor_id = phylogeny_df.at[idx, "ancestor_id"]
        if ancestor_id != idx:
            child_counts[ancestor_id] += 1

    sorted_ids = sorted(phylogeny_df["id"])
    n = len(sorted_ids)

    # Build offsets
    offset = 0
    start_map = {}
    for nid in sorted_ids:
        start_map[nid] = offset
        offset += child_counts[nid]

    # Scatter children
    csr_children = [0] * n
    insert_pos = dict(start_map)
    for idx in phylogeny_df.index:
        ancestor_id = phylogeny_df.at[idx, "ancestor_id"]
        if ancestor_id != idx:
            csr_children[insert_pos[ancestor_id]] = idx
            insert_pos[ancestor_id] += 1

    phylogeny_df[mark_as] = csr_children

    return phylogeny_df


def alifestd_mark_csr_children_asexual(
    phylogeny_df: pd.DataFrame,
    mutate: bool = False,
    *,
    mark_as: str = "csr_children",
) -> pd.DataFrame:
    """Add column `csr_children`, a flat array of child ids grouped by parent
    according to CSR offsets from the `csr_offsets` column.

    The output column name can be changed via the ``mark_as`` parameter.

    Entries are ordered so that node i's children occupy positions
    ``csr_offsets[i]`` to ``csr_offsets[i] + num_children[i]`` (exclusive).
    Entries beyond the total number of non-root nodes are unused.

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
        ancestor_ids = phylogeny_df["ancestor_id"].to_numpy()
        if "csr_offsets" in phylogeny_df.columns:
            csr_offsets = (
                phylogeny_df["csr_offsets"]
                .to_numpy()
                .astype(
                    np.int64,
                )
            )
        else:
            csr_offsets = _alifestd_mark_csr_offsets_asexual_fast_path(
                ancestor_ids,
            )
        phylogeny_df[mark_as] = _alifestd_mark_csr_children_asexual_fast_path(
            ancestor_ids.astype(np.int64),
            csr_offsets,
        )
        return phylogeny_df
    else:
        return _alifestd_mark_csr_children_asexual_slow_path(
            phylogeny_df,
            mark_as=mark_as,
        )


_raw_description = f"""{os.path.basename(__file__)} | (phyloframe v{get_phyloframe_version()}/joinem v{joinem.__version__})

Add column `csr_children`, a flat array of child ids grouped by parent according to CSR offsets.

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
        dfcli_module="phyloframe.legacy._alifestd_mark_csr_children_asexual",
        dfcli_version=get_phyloframe_version(),
    )
    parser.add_argument(
        "--mark-as",
        default="csr_children",
        type=str,
        help="output column name (default: csr_children)",
    )
    return parser


if __name__ == "__main__":
    begin_prod_logging()

    parser = _create_parser()
    args, __ = parser.parse_known_args()
    with log_context_duration(
        "phyloframe.legacy._alifestd_mark_csr_children_asexual",
        logging.info,
    ):
        _run_dataframe_cli(
            base_parser=parser,
            output_dataframe_op=delegate_polars_implementation()(
                functools.partial(
                    alifestd_mark_csr_children_asexual, mark_as=args.mark_as
                ),
            ),
        )
