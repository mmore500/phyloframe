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
from ._alifestd_mark_num_leaves_asexual import (
    _alifestd_mark_num_leaves_asexual_fast_path,
    alifestd_mark_num_leaves_asexual,
)
from ._alifestd_topological_sort import alifestd_topological_sort
from ._alifestd_try_add_ancestor_id_col import alifestd_try_add_ancestor_id_col


@jit(nopython=True)
def _alifestd_mark_num_mrca_leaf_pairs_asexual_fast_path(
    ancestor_ids: np.ndarray,
) -> np.ndarray:
    """Implementation detail for `alifestd_mark_num_mrca_leaf_pairs_asexual`
    and `alifestd_mark_num_mrca_leaf_pairs_polars`.

    Requires contiguous ids and topologically sorted rows.
    """
    num_leaves = _alifestd_mark_num_leaves_asexual_fast_path(ancestor_ids)
    num_leaves = num_leaves.astype(np.int64)

    # leaf pairs within each clade, less leaf pairs within each child clade
    num_pairs = num_leaves * (num_leaves - 1) // 2
    for idx, ancestor_id in enumerate(ancestor_ids):
        if ancestor_id != idx:  # exclude genesis cases
            num_child_leaves = num_leaves[idx]
            num_pairs[ancestor_id] -= (
                num_child_leaves * (num_child_leaves - 1) // 2
            )

    return num_pairs


def _alifestd_mark_num_mrca_leaf_pairs_asexual_slow_path(
    phylogeny_df: pd.DataFrame,
    mark_as: str,
) -> pd.DataFrame:
    """Implementation detail for `alifestd_mark_num_mrca_leaf_pairs_asexual`.

    Supports non-contiguous ids.
    """
    num_leaves = alifestd_mark_num_leaves_asexual(
        phylogeny_df, mark_as="num_leaves"
    ).set_index("id")["num_leaves"]
    num_leaves = num_leaves.astype(np.int64)
    num_pairs = num_leaves * (num_leaves - 1) // 2

    is_nonroot = (phylogeny_df["id"] != phylogeny_df["ancestor_id"]).to_numpy()
    child_pairs = (
        num_pairs.loc[phylogeny_df.loc[is_nonroot, "id"]]
        .groupby(phylogeny_df.loc[is_nonroot, "ancestor_id"].to_numpy())
        .sum()
    )
    num_pairs = num_pairs.sub(child_pairs, fill_value=0).astype(np.int64)

    phylogeny_df[mark_as] = num_pairs.loc[phylogeny_df["id"]].to_numpy()
    return phylogeny_df


def alifestd_mark_num_mrca_leaf_pairs_asexual(
    phylogeny_df: pd.DataFrame,
    mutate: bool = False,
    *,
    mark_as: str = "num_mrca_leaf_pairs",
) -> pd.DataFrame:
    """Add column `num_mrca_leaf_pairs` with count of unordered leaf pairs
    whose most recent common ancestor is each taxon.

    Computed as `C(n_v, 2) - sum_{u in children(v)} C(n_u, 2)`, where `n_v`
    is the number of leaves descending from `v`. Counts are zero for leaves
    and unifurcations, `n_L * n_R` for bifurcations, and sum to `C(N, 2)`
    within each tree with `N` leaves.

    The output column name can be changed via the ``mark_as`` parameter.

    A topological sort will be applied if `phylogeny_df` is not topologically
    sorted. Dataframe reindexing (e.g., df.index) may be applied.

    Input dataframe is not mutated by this operation unless `mutate` set True.
    If mutate set True, operation does not occur in place; still use return
    value to get transformed phylogeny dataframe.

    See Also
    --------
    alifestd_mark_num_mrca_leaf_pairs_polars :
        Polars-based implementation.
    alifestd_pool_chronological_consistency_asexual :
        Uses these counts as default weights.
    """

    if not mutate:
        phylogeny_df = phylogeny_df.copy()

    phylogeny_df = alifestd_try_add_ancestor_id_col(phylogeny_df, mutate=True)
    if "ancestor_id" not in phylogeny_df.columns:
        raise NotImplementedError(
            "alifestd_mark_num_mrca_leaf_pairs_asexual only supports "
            "asexual phylogenies.",
        )

    if phylogeny_df.empty:
        phylogeny_df[mark_as] = pd.Series(dtype=np.int64)
        return phylogeny_df

    if not alifestd_is_topologically_sorted(phylogeny_df):
        phylogeny_df = alifestd_topological_sort(phylogeny_df, mutate=True)

    if alifestd_has_contiguous_ids(phylogeny_df):
        phylogeny_df[
            mark_as
        ] = _alifestd_mark_num_mrca_leaf_pairs_asexual_fast_path(
            phylogeny_df["ancestor_id"].to_numpy().astype(np.int64),
        )
        return phylogeny_df
    else:
        return _alifestd_mark_num_mrca_leaf_pairs_asexual_slow_path(
            phylogeny_df,
            mark_as=mark_as,
        )


_raw_description = f"""{os.path.basename(__file__)} | (phyloframe v{get_phyloframe_version()}/joinem v{joinem.__version__})

Add column `num_mrca_leaf_pairs` with count of unordered leaf pairs whose most recent common ancestor is each taxon.

Data is assumed to be in alife standard format.

Additional Notes
================
- Use `--eager-read` if modifying data file inplace.

- This CLI entrypoint is experimental and may be subject to change.

See Also
========
phyloframe.legacy._alifestd_mark_num_mrca_leaf_pairs_polars :
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
        dfcli_module="phyloframe.legacy._alifestd_mark_num_mrca_leaf_pairs_asexual",
        dfcli_version=get_phyloframe_version(),
    )
    parser.add_argument(
        "--mark-as",
        default="num_mrca_leaf_pairs",
        type=str,
        help="output column name (default: num_mrca_leaf_pairs)",
    )
    return parser


if __name__ == "__main__":
    begin_prod_logging()

    parser = _create_parser()
    args, __ = parser.parse_known_args()
    with log_context_duration(
        "phyloframe.legacy._alifestd_mark_num_mrca_leaf_pairs_asexual",
        logging.info,
    ):
        _run_dataframe_cli(
            base_parser=parser,
            output_dataframe_op=delegate_polars_implementation()(
                functools.partial(
                    alifestd_mark_num_mrca_leaf_pairs_asexual,
                    mark_as=args.mark_as,
                ),
            ),
        )
