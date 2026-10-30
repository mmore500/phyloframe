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
from ._alifestd_try_add_ancestor_id_col import alifestd_try_add_ancestor_id_col


@jit(nopython=True)
def _alifestd_mark_min_child_value_asexual_fast_path(
    ancestor_ids: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    """Implementation detail for `alifestd_mark_min_child_value_asexual`."""
    result = np.copy(values)
    has_child = np.zeros(len(ancestor_ids), dtype=np.bool_)
    for k, anc in enumerate(ancestor_ids):
        if anc == k:
            continue
        if not has_child[anc] or values[k] < result[anc]:
            result[anc] = values[k]
            has_child[anc] = True
    return result


def alifestd_mark_min_child_value_asexual(
    phylogeny_df: pd.DataFrame,
    values: str,
    mutate: bool = False,
    *,
    mark_as: str = "min_child_value",
) -> pd.DataFrame:
    """Add column with minimum of ``values`` among each node's children.

    Nodes without children (i.e., leaves) keep their own value.

    The output column name can be changed via the ``mark_as`` parameter.
    NaN values are not supported; ``NotImplementedError`` is raised.

    Phylogeny must be asexual and have contiguous ids; otherwise
    ``NotImplementedError`` is raised.

    Input dataframe is not mutated by this operation unless ``mutate``
    is set True. If mutate set True, operation does not occur in place;
    still use return value to get transformed phylogeny dataframe.

    See Also
    --------
    alifestd_mark_min_child_value_polars :
        Polars-based implementation.
    alifestd_mark_lineage_cummin_asexual :
        Minimum along lineages, or over clades with ``reverse=True``.
    """
    if not mutate:
        phylogeny_df = phylogeny_df.copy()

    if values not in phylogeny_df.columns:
        raise ValueError(
            f"values column {values!r} not found in phylogeny_df",
        )

    phylogeny_df = alifestd_try_add_ancestor_id_col(phylogeny_df, mutate=True)
    if "ancestor_id" not in phylogeny_df.columns:
        raise NotImplementedError(
            "alifestd_mark_min_child_value_asexual only supports asexual "
            "phylogenies.",
        )

    if phylogeny_df.empty:
        phylogeny_df[mark_as] = phylogeny_df[values].astype(float)
        return phylogeny_df

    if not alifestd_has_contiguous_ids(phylogeny_df):
        raise NotImplementedError(
            "non-contiguous ids not supported",
        )

    values_arr = phylogeny_df[values].to_numpy(dtype=np.float64)
    if np.isnan(values_arr).any():
        raise NotImplementedError("NaN values not yet supported")

    phylogeny_df[mark_as] = _alifestd_mark_min_child_value_asexual_fast_path(
        phylogeny_df["ancestor_id"].to_numpy(),
        values_arr,
    )
    return phylogeny_df


_raw_description = f"""{os.path.basename(__file__)} | (phyloframe v{get_phyloframe_version()}/joinem v{joinem.__version__})

Add column with minimum of `values` among each node's children (own value for leaves).

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
        dfcli_module="phyloframe.legacy._alifestd_mark_min_child_value_asexual",
        dfcli_version=get_phyloframe_version(),
    )
    parser.add_argument(
        "--values",
        default=None,
        type=str,
        help="name of column to aggregate over children",
    )
    parser.add_argument(
        "--mark-as",
        default="min_child_value",
        type=str,
        help="output column name (default: min_child_value)",
    )
    return parser


if __name__ == "__main__":
    begin_prod_logging()

    parser = _create_parser()
    args, __ = parser.parse_known_args()
    with log_context_duration(
        "phyloframe.legacy._alifestd_mark_min_child_value_asexual",
        logging.info,
    ):
        _run_dataframe_cli(
            base_parser=parser,
            output_dataframe_op=delegate_polars_implementation()(
                functools.partial(
                    alifestd_mark_min_child_value_asexual,
                    values=args.values,
                    mark_as=args.mark_as,
                ),
            ),
        )
