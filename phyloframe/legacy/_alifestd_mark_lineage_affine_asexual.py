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
    scale: np.ndarray,
    offset: np.ndarray,
) -> np.ndarray:
    """Implementation detail for `alifestd_mark_lineage_affine_asexual`."""
    result = np.zeros(ancestor_ids.shape[0], dtype=np.float64)
    for k, anc in enumerate(ancestor_ids):
        not_root = anc != k
        result[k] = offset[k] + scale[k] * result[anc] * not_root
    return result


def alifestd_mark_lineage_affine_asexual(
    phylogeny_df: pd.DataFrame,
    scale: str,
    offset: str,
    mutate: bool = False,
    *,
    mark_as: str = "lineage_affine",
) -> pd.DataFrame:
    """Add column evaluating an affine recurrence down each lineage.

    The result at each node is ``scale * result[ancestor] + offset``,
    and ``offset`` at roots. For example, a unit ``scale`` gives the
    cumulative sum of ``offset`` along each lineage, and a zero ``offset``
    (except one at roots) gives the cumulative product of ``scale``.

    The output column name can be changed via the ``mark_as`` parameter.
    NaN values are not supported; ``NotImplementedError`` is raised.

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

    for col in (scale, offset):
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
        phylogeny_df[mark_as] = phylogeny_df[offset].astype(float)
        return phylogeny_df

    if not alifestd_has_contiguous_ids(phylogeny_df):
        raise NotImplementedError(
            "non-contiguous ids not supported",
        )

    if not alifestd_is_topologically_sorted(phylogeny_df):
        raise NotImplementedError(
            "non-topologically-sorted data not supported",
        )

    scale_arr = phylogeny_df[scale].to_numpy(dtype=np.float64)
    offset_arr = phylogeny_df[offset].to_numpy(dtype=np.float64)
    if np.isnan(scale_arr).any() or np.isnan(offset_arr).any():
        raise NotImplementedError("NaN values not yet supported")

    phylogeny_df[mark_as] = _alifestd_mark_lineage_affine_asexual_fast_path(
        phylogeny_df["ancestor_id"].to_numpy(),
        scale_arr,
        offset_arr,
    )
    return phylogeny_df


_raw_description = f"""{os.path.basename(__file__)} | (phyloframe v{get_phyloframe_version()}/joinem v{joinem.__version__})

Add column evaluating affine recurrence `scale * result[ancestor] + offset` down each lineage (`offset` at roots).

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
        "--scale",
        default=None,
        type=str,
        help="name of column multiplying ancestor's result",
    )
    parser.add_argument(
        "--offset",
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
                    scale=args.scale,
                    offset=args.offset,
                    mark_as=args.mark_as,
                ),
            ),
        )
