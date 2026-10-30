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
from ._alifestd_try_add_ancestor_id_col import alifestd_try_add_ancestor_id_col


@jit(nopython=True)
def _alifestd_mark_lineage_affine_asexual_fast_path(
    ancestor_ids: np.ndarray,
    multipliers: np.ndarray,
    addends: np.ndarray,
) -> np.ndarray:
    """Implementation detail for `alifestd_mark_lineage_affine_asexual`."""
    n = ancestor_ids.shape[0]
    result = np.empty(n, dtype=np.float64)
    for k in range(n):
        anc = ancestor_ids[k]
        if anc == k:
            result[k] = addends[k]
        else:
            result[k] = multipliers[k] * result[anc] + addends[k]
    return result


def alifestd_mark_lineage_affine_asexual(
    phylogeny_df: pd.DataFrame,
    multipliers: str,
    addends: str,
    mutate: bool = False,
    *,
    mark_as: str = "lineage_affine",
) -> pd.DataFrame:
    """Add column evaluating an affine recurrence down each lineage.

    The result at each node is ``multipliers * result[ancestor] + addends``,
    and ``addends`` at roots. For example, unit ``multipliers`` give the
    cumulative sum of ``addends`` along each lineage, and zero ``addends``
    (except ones at roots) give the cumulative product of ``multipliers``.

    The output column name can be changed via the ``mark_as`` parameter.
    NaN values propagate.

    Phylogeny must be asexual, topologically sorted, and have
    contiguous ids; otherwise ``NotImplementedError`` is raised.

    Input dataframe is not mutated by this operation unless ``mutate``
    is set True. If mutate set True, operation does not occur in place;
    still use return value to get transformed phylogeny dataframe.

    See Also
    --------
    alifestd_mark_lineage_affine_polars :
        Polars-based implementation.
    alifestd_mark_lineage_cumsum_asexual :
        Cumulative sum along each lineage.
    """
    if not mutate:
        phylogeny_df = phylogeny_df.copy()

    for col in (multipliers, addends):
        if col not in phylogeny_df.columns:
            raise ValueError(
                f"column {col!r} not found in phylogeny_df",
            )

    phylogeny_df = alifestd_try_add_ancestor_id_col(phylogeny_df, mutate=True)
    if "ancestor_id" not in phylogeny_df.columns:
        raise NotImplementedError(
            "alifestd_mark_lineage_affine_asexual only supports asexual "
            "phylogenies.",
        )

    if phylogeny_df.empty:
        phylogeny_df[mark_as] = phylogeny_df[addends].astype(float)
        return phylogeny_df

    if not alifestd_has_contiguous_ids(phylogeny_df):
        raise NotImplementedError(
            "non-contiguous ids not supported",
        )

    if not alifestd_is_topologically_sorted(phylogeny_df):
        raise NotImplementedError(
            "non-topologically-sorted data not supported",
        )

    phylogeny_df[mark_as] = _alifestd_mark_lineage_affine_asexual_fast_path(
        phylogeny_df["ancestor_id"].to_numpy(),
        phylogeny_df[multipliers].to_numpy(dtype=np.float64),
        phylogeny_df[addends].to_numpy(dtype=np.float64),
    )
    return phylogeny_df


_raw_description = f"""{os.path.basename(__file__)} | (phyloframe v{get_phyloframe_version()}/joinem v{joinem.__version__})

Add column evaluating affine recurrence `multipliers * result[ancestor] + addends` down each lineage (`addends` at roots).

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
        dfcli_module="phyloframe.legacy._alifestd_mark_lineage_affine_asexual",
        dfcli_version=get_phyloframe_version(),
    )
    parser.add_argument(
        "--multipliers",
        default=None,
        type=str,
        help="name of column multiplying ancestor's result",
    )
    parser.add_argument(
        "--addends",
        default=None,
        type=str,
        help="name of column added to multiplied ancestor's result",
    )
    parser.add_argument(
        "--mark-as",
        default="lineage_affine",
        type=str,
        help="output column name (default: lineage_affine)",
    )
    return parser


if __name__ == "__main__":
    begin_prod_logging()

    parser = _create_parser()
    args, __ = parser.parse_known_args()
    with log_context_duration(
        "phyloframe.legacy._alifestd_mark_lineage_affine_asexual",
        logging.info,
    ):
        _run_dataframe_cli(
            base_parser=parser,
            output_dataframe_op=delegate_polars_implementation()(
                functools.partial(
                    alifestd_mark_lineage_affine_asexual,
                    multipliers=args.multipliers,
                    addends=args.addends,
                    mark_as=args.mark_as,
                ),
            ),
        )
