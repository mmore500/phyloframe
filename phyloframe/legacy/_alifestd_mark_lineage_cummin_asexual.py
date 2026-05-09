import argparse
import functools
import logging
import os

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
from .._auxlib._jit import jit
from .._auxlib._log_context_duration import log_context_duration
from ._alifestd_has_contiguous_ids import alifestd_has_contiguous_ids
from ._alifestd_is_asexual import alifestd_is_asexual
from ._alifestd_is_topologically_sorted import alifestd_is_topologically_sorted
from ._alifestd_try_add_ancestor_id_col import alifestd_try_add_ancestor_id_col


@jit(nopython=True)
def _alifestd_mark_lineage_cummin_asexual_fast_path(
    ancestor_ids: np.ndarray,
    values: np.ndarray,
    reverse: bool,
) -> np.ndarray:
    """Implementation detail for `alifestd_mark_lineage_cummin_asexual`."""
    n = ancestor_ids.shape[0]
    result = np.copy(values)
    for k in range(n - 1, -1, -1) if reverse else range(n):
        anc = ancestor_ids[k]
        if anc == k:
            continue
        # NaN propagation: NaN != NaN is True, < returns False for NaN
        if reverse:
            s = result[k]
            if s != s or s < result[anc]:
                result[anc] = s
        else:
            s = result[anc]
            if s != s or s < result[k]:
                result[k] = s
    return result


def alifestd_mark_lineage_cummin_asexual(
    phylogeny_df: pd.DataFrame,
    values: str,
    mutate: bool = False,
    *,
    mark_as: str = "lineage_cummin",
    reverse: bool = False,
    skipna: bool = True,
) -> pd.DataFrame:
    """Add column with minimum of ``values`` along each lineage.

    With ``reverse=False`` (default), the result at each node is the
    minimum of ``values`` along the path from the root to that node,
    inclusive. With ``reverse=True``, the result at each node is the
    minimum of ``values`` over the entire clade rooted at that node,
    inclusive.

    The output column name can be changed via the ``mark_as`` parameter.
    NaN values are treated as +inf if ``skipna`` (default), else
    propagate.

    Phylogeny must be asexual, topologically sorted, and have
    contiguous ids; otherwise ``NotImplementedError`` is raised.

    Input dataframe is not mutated by this operation unless ``mutate``
    is set True. If mutate set True, operation does not occur in place;
    still use return value to get transformed phylogeny dataframe.
    """
    if not mutate:
        phylogeny_df = phylogeny_df.copy()

    if not alifestd_is_asexual(phylogeny_df):
        raise NotImplementedError(
            "alifestd_mark_lineage_cummin_asexual only supports asexual "
            "phylogenies.",
        )

    if values not in phylogeny_df.columns:
        raise ValueError(
            f"values column {values!r} not found in phylogeny_df",
        )

    phylogeny_df = alifestd_try_add_ancestor_id_col(phylogeny_df, mutate=True)

    if phylogeny_df.empty:
        phylogeny_df[mark_as] = phylogeny_df[values].copy()
        return phylogeny_df

    if not alifestd_has_contiguous_ids(phylogeny_df):
        raise NotImplementedError(
            "non-contiguous ids not supported",
        )

    if not alifestd_is_topologically_sorted(phylogeny_df):
        raise NotImplementedError(
            "non-topologically-sorted data not supported",
        )

    values_arr = phylogeny_df[values].to_numpy()
    if skipna and np.issubdtype(values_arr.dtype, np.floating):
        values_arr = np.where(np.isnan(values_arr), np.inf, values_arr)

    phylogeny_df[mark_as] = _alifestd_mark_lineage_cummin_asexual_fast_path(
        phylogeny_df["ancestor_id"].to_numpy(dtype=np.uint64),
        values_arr,
        reverse,
    )
    return phylogeny_df


_raw_description = f"""{os.path.basename(__file__)} | (phyloframe v{get_phyloframe_version()}/joinem v{joinem.__version__})

Add column with minimum of `values` along each lineage.

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
        dfcli_module="phyloframe.legacy._alifestd_mark_lineage_cummin_asexual",
        dfcli_version=get_phyloframe_version(),
    )
    parser.add_argument(
        "--values",
        default=None,
        type=str,
        help="name of column to aggregate along lineage",
    )
    parser.add_argument(
        "--mark-as",
        default="lineage_cummin",
        type=str,
        help="output column name (default: lineage_cummin)",
    )
    add_bool_arg(
        parser,
        "reverse",
        default=False,
        help="aggregate over clade rooted at each node "
        "instead of along lineage from root",
    )
    add_bool_arg(
        parser,
        "skipna",
        default=True,
        help="treat NaN values as identity; "
        "use --no-skipna to propagate NaN instead",
    )
    return parser


if __name__ == "__main__":
    begin_prod_logging()

    parser = _create_parser()
    args, __ = parser.parse_known_args()
    with log_context_duration(
        "phyloframe.legacy._alifestd_mark_lineage_cummin_asexual",
        logging.info,
    ):
        _run_dataframe_cli(
            base_parser=parser,
            output_dataframe_op=delegate_polars_implementation()(
                functools.partial(
                    alifestd_mark_lineage_cummin_asexual,
                    values=args.values,
                    mark_as=args.mark_as,
                    reverse=args.reverse,
                    skipna=args.skipna,
                ),
            ),
        )
