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
from .._auxlib._log_context_duration import log_context_duration
from ._alifestd_is_working_format_asexual import (
    alifestd_is_working_format_asexual,
)
from ._alifestd_mark_lineage_cummax_asexual import (
    _alifestd_mark_lineage_cummax_asexual_fast_path,
)
from ._alifestd_mark_num_children_asexual import (
    _alifestd_mark_num_children_asexual_fast_path,
)
from ._alifestd_try_add_ancestor_id_col import alifestd_try_add_ancestor_id_col


def _alifestd_mark_stem_node_id_asexual_fast_path(
    ancestor_ids: np.ndarray,
) -> np.ndarray:
    """Implementation detail for `alifestd_mark_stem_node_id_asexual`."""
    ids = np.arange(len(ancestor_ids))
    is_branching = _alifestd_mark_num_children_asexual_fast_path(ancestor_ids)
    is_branching = is_branching >= 2
    # ids increase along each lineage, so the running max of branching ids
    # (else -1) is the nearest branching node at or above each node
    nearest_branching_ids = _alifestd_mark_lineage_cummax_asexual_fast_path(
        ancestor_ids, np.where(is_branching, ids, -1), False, False
    )
    stem_node_ids = nearest_branching_ids[ancestor_ids]
    is_root = ancestor_ids == ids
    return np.where(is_root | (stem_node_ids < 0), ids, stem_node_ids)


def alifestd_mark_stem_node_id_asexual(
    phylogeny_df: pd.DataFrame,
    mutate: bool = False,
    *,
    mark_as: str = "stem_node_id",
) -> pd.DataFrame:
    """Add column `stem_node_id`, the id of each node's stem node: its
    nearest strict ancestor with at least two children, where its lineage
    splits from its sister lineages, bypassing unifurcations.

    A node without one (e.g., a root, or a node below only unifurcations)
    is its own stem node.

    The output column name can be changed via the ``mark_as`` parameter.

    Phylogeny must be asexual and in working format (contiguous,
    topologically sorted ids; see `alifestd_to_working_format`); otherwise
    ``NotImplementedError`` is raised.

    Input dataframe is not mutated by this operation unless ``mutate``
    is set True. If mutate set True, operation does not occur in place;
    still use return value to get transformed phylogeny dataframe.

    See Also
    --------
    alifestd_mark_stem_node_id_polars :
        Polars-based implementation.
    """
    if not mutate:
        phylogeny_df = phylogeny_df.copy()

    phylogeny_df = alifestd_try_add_ancestor_id_col(phylogeny_df, mutate=True)
    if "ancestor_id" not in phylogeny_df.columns:
        raise NotImplementedError(
            "alifestd_mark_stem_node_id_asexual only supports asexual "
            "phylogenies.",
        )

    if phylogeny_df.empty:
        phylogeny_df[mark_as] = phylogeny_df["id"].copy()
        return phylogeny_df

    if not alifestd_is_working_format_asexual(phylogeny_df, mutate=True):
        raise NotImplementedError(
            "current implementation requires phylogeny_df in working format",
        )

    phylogeny_df[mark_as] = _alifestd_mark_stem_node_id_asexual_fast_path(
        phylogeny_df["ancestor_id"].to_numpy(),
    )
    return phylogeny_df


_raw_description = f"""{os.path.basename(__file__)} | (phyloframe v{get_phyloframe_version()}/joinem v{joinem.__version__})

Add column `stem_node_id`, the id of each node's nearest strict ancestor with
at least two children, bypassing unifurcations (else, the node's own id).

Data is assumed to be in alife standard format.

Additional Notes
================
- Requires contiguous, topologically sorted ids.

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
        dfcli_module="phyloframe.legacy._alifestd_mark_stem_node_id_asexual",
        dfcli_version=get_phyloframe_version(),
    )
    parser.add_argument(
        "--mark-as",
        default="stem_node_id",
        type=str,
        help="output column name (default: stem_node_id)",
    )
    return parser


if __name__ == "__main__":
    begin_prod_logging()

    parser = _create_parser()
    args, __ = parser.parse_known_args()
    with log_context_duration(
        "phyloframe.legacy._alifestd_mark_stem_node_id_asexual",
        logging.info,
    ):
        _run_dataframe_cli(
            base_parser=parser,
            output_dataframe_op=delegate_polars_implementation()(
                functools.partial(
                    alifestd_mark_stem_node_id_asexual,
                    mark_as=args.mark_as,
                ),
            ),
        )
