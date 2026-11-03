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
def _alifestd_mark_lineage_ffill_asexual_fast_path(
    ancestor_ids: np.ndarray,
    is_missing: np.ndarray,
) -> np.ndarray:
    """Implementation detail for `alifestd_mark_lineage_ffill_asexual` and
    `alifestd_mark_lineage_ffill_polars`.

    Requires contiguous ids and topologically sorted rows. Returns, for each
    node, the id of the node to take its value from: itself if not missing,
    else its nearest ancestor that is not missing. Missing nodes with no such
    ancestor take from their lineage's root, so remain missing.
    """
    sources = np.arange(len(ancestor_ids))
    for idx, ancestor_id in enumerate(ancestor_ids):
        if is_missing[idx]:  # roots are own ancestor, so keep own source
            sources[idx] = sources[ancestor_id]
    return sources


def alifestd_mark_lineage_ffill_asexual(
    phylogeny_df: pd.DataFrame,
    values: str,
    mutate: bool = False,
    *,
    mark_as: str = "lineage_ffill",
) -> pd.DataFrame:
    """Add column with missing ``values`` filled from nearest ancestor with a
    non-missing value.

    Non-missing values are unchanged. Missing values with no non-missing
    ancestor (e.g., missing roots) remain missing. Values are considered
    missing per `pandas.isna` (e.g., NaN, None, or NA). Column dtype is
    preserved.

    The output column name can be changed via the ``mark_as`` parameter.

    Input dataframe is not mutated by this operation unless ``mutate``
    is set True. If mutate set True, operation does not occur in place;
    still use return value to get transformed phylogeny dataframe.

    Raises
    ------
    NotImplementedError
        If `phylogeny_df` is not asexual, has non-contiguous ids, or is not
        topologically sorted.

    See Also
    --------
    alifestd_mark_lineage_ffill_polars :
        Polars-based implementation.
    alifestd_mark_lineage_cummax_asexual :
        Maximum along lineages.
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
            "alifestd_mark_lineage_ffill_asexual only supports asexual "
            "phylogenies.",
        )

    if phylogeny_df.empty:
        phylogeny_df[mark_as] = phylogeny_df[values].copy()
        return phylogeny_df

    if not alifestd_has_contiguous_ids(phylogeny_df):
        raise NotImplementedError("non-contiguous ids not supported")

    if not alifestd_is_topologically_sorted(phylogeny_df):
        raise NotImplementedError(
            "non-topologically-sorted data not supported",
        )

    sources = _alifestd_mark_lineage_ffill_asexual_fast_path(
        phylogeny_df["ancestor_id"].to_numpy(),
        phylogeny_df[values].isna().to_numpy(),
    )
    # sources are row positions; take .array to skip index alignment
    phylogeny_df[mark_as] = phylogeny_df[values].iloc[sources].array
    return phylogeny_df


_raw_description = f"""{os.path.basename(__file__)} | (phyloframe v{get_phyloframe_version()}/joinem v{joinem.__version__})

Add column with missing `values` filled from nearest ancestor with a non-missing value.

Data is assumed to be in alife standard format.

Additional Notes
================
- Requires asexual phylogeny with contiguous ids and topologically sorted rows.

- Missing values with no non-missing ancestor (e.g., missing roots) remain missing.

- Use `--eager-read` if modifying data file inplace.

- This CLI entrypoint is experimental and may be subject to change.

See Also
========
phyloframe.legacy._alifestd_mark_lineage_ffill_polars :
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
        dfcli_module="phyloframe.legacy._alifestd_mark_lineage_ffill_asexual",
        dfcli_version=get_phyloframe_version(),
    )
    parser.add_argument(
        "--values",
        default=None,
        type=str,
        help="name of column to fill along lineage",
    )
    parser.add_argument(
        "--mark-as",
        default="lineage_ffill",
        type=str,
        help="output column name (default: lineage_ffill)",
    )
    return parser


if __name__ == "__main__":
    begin_prod_logging()

    parser = _create_parser()
    args, __ = parser.parse_known_args()
    with log_context_duration(
        "phyloframe.legacy._alifestd_mark_lineage_ffill_asexual",
        logging.info,
    ):
        _run_dataframe_cli(
            base_parser=parser,
            output_dataframe_op=delegate_polars_implementation()(
                functools.partial(
                    alifestd_mark_lineage_ffill_asexual,
                    values=args.values,
                    mark_as=args.mark_as,
                ),
            ),
        )
