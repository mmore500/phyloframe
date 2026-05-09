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
from ._alifestd_is_asexual import alifestd_is_asexual
from ._alifestd_is_topologically_sorted import alifestd_is_topologically_sorted
from ._alifestd_try_add_ancestor_id_col import alifestd_try_add_ancestor_id_col

# Op codes used by the shared jitted helper.
_OP_SUM = 0
_OP_PROD = 1
_OP_MIN = 2
_OP_MAX = 3


@jit(nopython=True)
def _alifestd_mark_lineage_op_asexual_fast_path(
    ancestor_ids: np.ndarray,
    values: np.ndarray,
    op_code: int,
    reverse: bool,
    skipna: bool,
) -> np.ndarray:
    """Compute lineage cumulative aggregation along contiguous-id, sorted ids.

    Parameters
    ----------
    ancestor_ids
        Ancestor id for each row, indexed by contiguous id.
    values
        Per-node values to aggregate; cast to float64 by caller.
    op_code
        0 sum, 1 prod, 2 min, 3 max.
    reverse
        If False, propagate from root to descendants (lineage prefix
        aggregate from root inclusive). If True, propagate from
        descendants to ancestors (clade-rooted aggregate inclusive).
    skipna
        If True, NaN entries in ``values`` are treated as the
        identity element for ``op_code``. If False, NaN propagates.
    """
    n = ancestor_ids.shape[0]
    result = np.empty(n, dtype=np.float64)

    # initialize with own values, replacing NaN with identity if skipna
    for i in range(n):
        v = values[i]
        if skipna and np.isnan(v):
            if op_code == _OP_SUM:
                v = 0.0
            elif op_code == _OP_PROD:
                v = 1.0
            elif op_code == _OP_MIN:
                v = np.inf
            else:
                v = -np.inf
        result[i] = v

    if not reverse:
        # forward: result[i] combines result[ancestor] with own value
        for i in range(n):
            anc = ancestor_ids[i]
            if anc == i:
                continue
            a = result[anc]
            v = result[i]
            if op_code == _OP_SUM:
                result[i] = a + v
            elif op_code == _OP_PROD:
                result[i] = a * v
            elif op_code == _OP_MIN:
                if a < v:
                    result[i] = a
            else:
                if a > v:
                    result[i] = a
    else:
        # reverse: each ancestor aggregates over its subtree
        for j in range(n - 1, -1, -1):
            anc = ancestor_ids[j]
            if anc == j:
                continue
            a = result[anc]
            v = result[j]
            if op_code == _OP_SUM:
                result[anc] = a + v
            elif op_code == _OP_PROD:
                result[anc] = a * v
            elif op_code == _OP_MIN:
                if v < a:
                    result[anc] = v
            else:
                if v > a:
                    result[anc] = v

    return result


def _alifestd_mark_lineage_op_asexual(
    phylogeny_df: pd.DataFrame,
    values: str,
    op_code: int,
    *,
    mutate: bool,
    mark_as: str,
    reverse: bool,
    skipna: bool,
) -> pd.DataFrame:
    """Shared implementation for all lineage op marks."""
    if not mutate:
        phylogeny_df = phylogeny_df.copy()

    if not alifestd_is_asexual(phylogeny_df):
        raise NotImplementedError(
            "alifestd_mark_lineage_*_asexual only supports asexual "
            "phylogenies.",
        )

    if values not in phylogeny_df.columns:
        raise ValueError(
            f"values column {values!r} not found in phylogeny_df",
        )

    phylogeny_df = alifestd_try_add_ancestor_id_col(phylogeny_df, mutate=True)

    if phylogeny_df.empty:
        phylogeny_df[mark_as] = np.empty(0, dtype=np.float64)
        return phylogeny_df

    if not alifestd_has_contiguous_ids(phylogeny_df):
        raise NotImplementedError(
            "non-contiguous ids not supported",
        )

    if not alifestd_is_topologically_sorted(phylogeny_df):
        raise NotImplementedError(
            "non-topologically-sorted data not supported",
        )

    phylogeny_df[mark_as] = _alifestd_mark_lineage_op_asexual_fast_path(
        phylogeny_df["ancestor_id"].to_numpy(dtype=np.uint64),
        phylogeny_df[values].to_numpy(dtype=np.float64),
        op_code,
        reverse,
        skipna,
    )
    return phylogeny_df


def alifestd_mark_lineage_cumsum_asexual(
    phylogeny_df: pd.DataFrame,
    values: str,
    mutate: bool = False,
    *,
    mark_as: str = "lineage_cumsum",
    reverse: bool = False,
    skipna: bool = True,
) -> pd.DataFrame:
    """Add column with cumulative sum of ``values`` along each lineage.

    With ``reverse=False`` (default), the result at each node is the
    sum of ``values`` along the path from the root to that node,
    inclusive. With ``reverse=True``, the result at each node is the
    sum of ``values`` over the entire clade rooted at that node,
    inclusive.

    The output column name can be changed via the ``mark_as`` parameter.
    NaN values are treated as 0 if ``skipna`` (default), else propagate.

    Phylogeny must be asexual, topologically sorted, and have
    contiguous ids; otherwise ``NotImplementedError`` is raised.

    Input dataframe is not mutated by this operation unless ``mutate``
    is set True. If mutate set True, operation does not occur in place;
    still use return value to get transformed phylogeny dataframe.
    """
    return _alifestd_mark_lineage_op_asexual(
        phylogeny_df,
        values,
        _OP_SUM,
        mutate=mutate,
        mark_as=mark_as,
        reverse=reverse,
        skipna=skipna,
    )


_raw_description = f"""{os.path.basename(__file__)} | (phyloframe v{get_phyloframe_version()}/joinem v{joinem.__version__})

Add column with cumulative sum of `values` along each lineage.

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
        dfcli_module="phyloframe.legacy._alifestd_mark_lineage_cumsum_asexual",
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
        default="lineage_cumsum",
        type=str,
        help="output column name (default: lineage_cumsum)",
    )
    parser.add_argument(
        "--reverse",
        action="store_true",
        help="aggregate over clade rooted at each node "
        "instead of along lineage from root",
    )
    parser.add_argument(
        "--no-skipna",
        action="store_true",
        help="propagate NaN values rather than treating as identity",
    )
    return parser


if __name__ == "__main__":
    begin_prod_logging()

    parser = _create_parser()
    args, __ = parser.parse_known_args()
    with log_context_duration(
        "phyloframe.legacy._alifestd_mark_lineage_cumsum_asexual",
        logging.info,
    ):
        _run_dataframe_cli(
            base_parser=parser,
            output_dataframe_op=delegate_polars_implementation()(
                functools.partial(
                    alifestd_mark_lineage_cumsum_asexual,
                    values=args.values,
                    mark_as=args.mark_as,
                    reverse=args.reverse,
                    skipna=not args.no_skipna,
                ),
            ),
        )
