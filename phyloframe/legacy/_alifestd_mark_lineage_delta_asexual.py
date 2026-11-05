import argparse
import functools
import logging
import os
import typing
import warnings

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
from ._alifestd_try_add_ancestor_id_col import alifestd_try_add_ancestor_id_col


def _alifestd_mark_lineage_delta_asexual_fast_path(
    ancestor_ids: np.ndarray,
    values: np.ndarray,
    basal_value: typing.Optional[typing.Any] = None,
) -> np.ndarray:
    """Implementation detail for `alifestd_mark_lineage_delta_asexual`."""
    ancestor_ids = np.asarray(ancestor_ids, dtype=np.int64)
    values = np.asarray(values)
    if values.dtype == np.bool_:
        values = values.astype(np.int8)  # deltas -1, 0, or 1 fit
    # a root is its own ancestor, so its delta is 0, unless measured from a
    # basal value
    ancestral_values = values[ancestor_ids]
    if basal_value is not None:
        is_root = ancestor_ids == np.arange(len(ancestor_ids))
        ancestral_values = np.where(
            is_root,
            np.asarray(basal_value).astype(values.dtype),
            ancestral_values,
        )
    if (
        np.issubdtype(values.dtype, np.unsignedinteger)
        and (ancestral_values > values).any()
    ):
        warnings.warn(
            f"negative deltas of unsigned values ({values.dtype}) wrap around",
        )
    return (values - ancestral_values).astype(values.dtype)


def alifestd_mark_lineage_delta_asexual(
    phylogeny_df: pd.DataFrame,
    values: str,
    mutate: bool = False,
    *,
    basal_value: typing.Optional[typing.Any] = None,
    mark_as: str = "lineage_delta",
) -> pd.DataFrame:
    """Add column with the change in ``values`` from each node's parent.

    The result at each node is its value minus its parent's value. A
    root's delta is 0, or, if ``basal_value`` is given, the root's value
    minus ``basal_value`` (e.g., with ``basal_value=0``, the root's own
    value). NaN values propagate to a node and its children.

    Deltas take the dtype of ``values``, so integer deltas can overflow, and
    negative deltas of unsigned values may wrap; cast to a wider or signed
    dtype first to avoid this. Boolean values are cast to int8, producing
    deltas: +1 for a gain, -1 for a loss, and 0 for no change.

    The output column name can be changed via the ``mark_as`` parameter.

    Phylogeny must be asexual and have contiguous ids; otherwise
    ``NotImplementedError`` is raised.

    Input dataframe is not mutated by this operation unless ``mutate``
    is set True. If mutate set True, operation does not occur in place;
    still use return value to get transformed phylogeny dataframe.

    See Also
    --------
    alifestd_mark_lineage_delta_polars :
        Polars-based implementation.
    alifestd_mark_lineage_cumsum_asexual :
        Sums values along each lineage.
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
            "alifestd_mark_lineage_delta_asexual only supports asexual "
            "phylogenies.",
        )

    if phylogeny_df.empty:
        phylogeny_df[mark_as] = phylogeny_df[values].copy()
        return phylogeny_df

    if not alifestd_has_contiguous_ids(phylogeny_df):
        raise NotImplementedError(
            "non-contiguous ids not supported",
        )

    phylogeny_df[mark_as] = _alifestd_mark_lineage_delta_asexual_fast_path(
        phylogeny_df["ancestor_id"].to_numpy(),
        phylogeny_df[values].to_numpy(),
        basal_value,
    )
    return phylogeny_df


_raw_description = f"""{os.path.basename(__file__)} | (phyloframe v{get_phyloframe_version()}/joinem v{joinem.__version__})

Add column with the change in `values` from each node's parent. A root's
delta is 0, or its value minus `--basal-value` if given.

Data is assumed to be in alife standard format.

Additional Notes
================
- Requires contiguous ids.

- Boolean values are taken as 0 and 1.

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
        dfcli_module="phyloframe.legacy._alifestd_mark_lineage_delta_asexual",
        dfcli_version=get_phyloframe_version(),
    )
    parser.add_argument(
        "--values",
        default=None,
        type=str,
        help="name of column to take deltas of",
    )
    parser.add_argument(
        "--basal-value",
        default=None,
        type=float,
        help=(
            "value that each root's delta is measured from (default: "
            "none, so each root's delta is 0)"
        ),
    )
    parser.add_argument(
        "--mark-as",
        default="lineage_delta",
        type=str,
        help="output column name (default: lineage_delta)",
    )
    return parser


if __name__ == "__main__":
    begin_prod_logging()

    parser = _create_parser()
    args, __ = parser.parse_known_args()
    with log_context_duration(
        "phyloframe.legacy._alifestd_mark_lineage_delta_asexual",
        logging.info,
    ):
        _run_dataframe_cli(
            base_parser=parser,
            output_dataframe_op=delegate_polars_implementation()(
                functools.partial(
                    alifestd_mark_lineage_delta_asexual,
                    values=args.values,
                    basal_value=args.basal_value,
                    mark_as=args.mark_as,
                ),
            ),
        )
