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
from ._alifestd_has_contiguous_ids import alifestd_has_contiguous_ids
from ._alifestd_is_topologically_sorted import alifestd_is_topologically_sorted
from ._alifestd_mark_lineage_cummax_asexual import (
    _alifestd_mark_lineage_cummax_asexual_fast_path,
)
from ._alifestd_mark_lineage_cumsum_asexual import (
    _alifestd_mark_lineage_cumsum_asexual_fast_path,
)
from ._alifestd_mark_num_children_asexual import (
    _alifestd_mark_num_children_asexual_fast_path,
)
from ._alifestd_try_add_ancestor_id_col import alifestd_try_add_ancestor_id_col


def _alifestd_mark_unifurcation_length_asexual_fast_path(
    ancestor_ids: np.ndarray,
    weights: typing.Optional[np.ndarray] = None,
) -> np.ndarray:
    """Implementation detail for `alifestd_mark_unifurcation_length_asexual`.

    Assumes contiguous ids and topological sorting.
    """
    ancestor_ids = np.asarray(ancestor_ids, dtype=np.int64)
    ids = np.arange(len(ancestor_ids))
    is_root = ancestor_ids == ids
    num_children = _alifestd_mark_num_children_asexual_fast_path(ancestor_ids)
    # roots are not unifurcations, as in alifestd_collapse_unifurcations
    is_unifurcation = (num_children == 1) & ~is_root

    if weights is None:
        weights = np.ones(len(ancestor_ids), dtype=np.int64)
    unifurcation_weights = np.where(is_unifurcation, weights, 0)

    # total weight of unifurcations along lineage from root, inclusive
    cumsums = _alifestd_mark_lineage_cumsum_asexual_fast_path(
        ancestor_ids, unifurcation_weights, False
    )
    # nearest non-unifurcation along lineage, inclusive; roots always count
    anchor_ids = _alifestd_mark_lineage_cummax_asexual_fast_path(
        ancestor_ids, np.where(is_unifurcation, -1, ids), False
    )

    # weight of contiguous unifurcations above parent through its anchor
    return np.where(
        is_root,
        0,
        cumsums[ancestor_ids] - cumsums[anchor_ids[ancestor_ids]],
    )


def alifestd_mark_unifurcation_length_asexual(
    phylogeny_df: pd.DataFrame,
    mutate: bool = False,
    *,
    mark_as: str = "unifurcation_length",
    weights: typing.Optional[str] = None,
) -> pd.DataFrame:
    """Add column `unifurcation_length`, measuring for each node the run of
    contiguous unifurcations (nodes with exactly one child) directly above
    it, i.e., the ancestors that `alifestd_collapse_unifurcations` would
    bypass.

    By default, length counts unifurcating nodes. If ``weights`` names a
    column, length sums that column's values over those nodes instead
    (e.g., to measure length in branch length units, or to accumulate
    counts across repeated unifurcation collapses).

    As in `alifestd_collapse_unifurcations`, roots are not considered
    unifurcations. The output column name can be changed via the
    ``mark_as`` parameter.

    Phylogeny must be asexual, topologically sorted, and have
    contiguous ids; otherwise ``NotImplementedError`` is raised.

    Input dataframe is not mutated by this operation unless ``mutate``
    is set True. If mutate set True, operation does not occur in place;
    still use return value to get transformed phylogeny dataframe.

    See Also
    --------
    alifestd_mark_unifurcation_length_polars :
        Polars-based implementation.
    alifestd_collapse_unifurcations :
        Removes the unifurcations measured here.
    """
    if not mutate:
        phylogeny_df = phylogeny_df.copy()

    if weights is not None and weights not in phylogeny_df.columns:
        raise ValueError(
            f"weights column {weights!r} not found in phylogeny_df",
        )

    phylogeny_df = alifestd_try_add_ancestor_id_col(phylogeny_df, mutate=True)
    if "ancestor_id" not in phylogeny_df.columns:
        raise NotImplementedError(
            "alifestd_mark_unifurcation_length_asexual only supports "
            "asexual phylogenies.",
        )

    if phylogeny_df.empty:
        phylogeny_df[mark_as] = (
            0 if weights is None else phylogeny_df[weights].copy()
        )
        return phylogeny_df

    if not alifestd_has_contiguous_ids(phylogeny_df):
        raise NotImplementedError(
            "non-contiguous ids not supported",
        )

    if not alifestd_is_topologically_sorted(phylogeny_df):
        raise NotImplementedError(
            "non-topologically-sorted data not supported",
        )

    phylogeny_df[
        mark_as
    ] = _alifestd_mark_unifurcation_length_asexual_fast_path(
        phylogeny_df["ancestor_id"].to_numpy(),
        None if weights is None else phylogeny_df[weights].to_numpy(),
    )
    return phylogeny_df


_raw_description = f"""{os.path.basename(__file__)} | (phyloframe v{get_phyloframe_version()}/joinem v{joinem.__version__})

Add column `unifurcation_length`, measuring for each node the run of contiguous unifurcations directly above it (i.e., ancestors that unifurcation collapse would bypass).

Data is assumed to be in alife standard format.

Additional Notes
================
- Use `--weights` to sum a column over unifurcations, instead of counting them.

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
        dfcli_module="phyloframe.legacy._alifestd_mark_unifurcation_length_asexual",
        dfcli_version=get_phyloframe_version(),
    )
    parser.add_argument(
        "--mark-as",
        default="unifurcation_length",
        type=str,
        help="output column name (default: unifurcation_length)",
    )
    parser.add_argument(
        "--weights",
        default=None,
        type=str,
        help="column to sum over unifurcations (default: None, count them)",
    )
    return parser


if __name__ == "__main__":
    begin_prod_logging()

    parser = _create_parser()
    args, __ = parser.parse_known_args()
    with log_context_duration(
        "phyloframe.legacy._alifestd_mark_unifurcation_length_asexual",
        logging.info,
    ):
        _run_dataframe_cli(
            base_parser=parser,
            output_dataframe_op=delegate_polars_implementation()(
                functools.partial(
                    alifestd_mark_unifurcation_length_asexual,
                    mark_as=args.mark_as,
                    weights=args.weights,
                ),
            ),
        )
