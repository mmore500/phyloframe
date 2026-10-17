import argparse
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


def alifestd_unminify_asexual(
    phylogeny_df: pd.DataFrame,
    dtype_id: typing.Optional[type] = np.int64,
    mutate: bool = False,
) -> pd.DataFrame:
    """Restore alife standard format from a phylogeny minified by
    `alifestd_minify_asexual`.

    Replaces the `ancestor_offset` column with an `ancestor_id` column and
    adds an `id` column corresponding to row position. Row order is
    preserved, so the returned phylogeny will have contiguous ids and be
    topologically sorted.

    The `id` and `ancestor_id` columns will have dtype `dtype_id`. If
    `dtype_id` is None, the smallest signed integer dtype that can hold all
    ids is chosen automatically based on row count. Any existing `id` or
    `ancestor_id` columns will be overwritten.

    No `ancestor_list` column is added. To add one, call
    `alifestd_try_add_ancestor_list_col` on the result.

    Input dataframe is not mutated by this operation unless `mutate` set True.
    If mutate set True, operation does not occur in place; still use return
    value to get transformed phylogeny dataframe.

    See Also
    --------
    alifestd_unminify_polars :
        Polars-based implementation.
    alifestd_minify_asexual :
        Inverse operation, producing minified format.
    alifestd_try_add_ancestor_list_col :
        Add alife standard `ancestor_list` column.
    """
    for column in "id", "ancestor_id":
        if column in phylogeny_df.columns:
            warnings.warn(
                f"alifestd_unminify_asexual overwrites existing {column}",
            )

    if not mutate:
        phylogeny_df = phylogeny_df.copy()

    if dtype_id is None:
        dtype_id = np.min_scalar_type(-(len(phylogeny_df) or 1))
        logging.info(
            "- alifestd_unminify_asexual: "
            f"detected {len(phylogeny_df)} rows, chose id dtype {dtype_id}",
        )

    ids = np.arange(len(phylogeny_df), dtype=dtype_id)
    ancestor_ids = ids - phylogeny_df["ancestor_offset"].to_numpy(
        dtype=dtype_id,
    )

    phylogeny_df = phylogeny_df.drop(
        columns=["id", "ancestor_id", "ancestor_offset"],
        errors="ignore",
    )
    phylogeny_df.insert(0, "id", ids)
    phylogeny_df.insert(1, "ancestor_id", ancestor_ids)
    return phylogeny_df


_raw_description = f"""{os.path.basename(__file__)} | (phyloframe v{get_phyloframe_version()}/joinem v{joinem.__version__})

Restore alife standard format from a phylogeny minified by `alifestd_minify_asexual`.

Replaces the `ancestor_offset` column with an `ancestor_id` column and adds an `id` column corresponding to row position.
Row order is preserved, so the output phylogeny will have contiguous ids and be topologically sorted.

No `ancestor_list` column is added.
To add one, chain with `alifestd_try_add_ancestor_list_col` using the `phyloframe.legacy._alifestd_pipe_unary_ops` entrypoint, e.g.,
    ls -1 "minified.pqt" \\
      | python3 -m phyloframe.legacy._alifestd_pipe_unary_ops \\
        --op "pfl.alifestd_unminify_asexual" \\
        --op "pfl.alifestd_try_add_ancestor_list_col" \\
        "output.pqt"

Data is assumed to be in minified format.

Additional Notes
================
- Use `--eager-read` if modifying data file inplace.

- This CLI entrypoint is experimental and may be subject to change.

See Also
========
phyloframe.legacy._alifestd_unminify_polars :
    Entrypoint for high-performance Polars-based implementation.
phyloframe.legacy._alifestd_minify_asexual :
    CLI entrypoint for inverse operation, producing minified format.
phyloframe.legacy._alifestd_try_add_ancestor_list_col :
    CLI entrypoint to add alife standard `ancestor_list` column.
phyloframe.legacy._alifestd_pipe_unary_ops :
    CLI entrypoint to chain multiple operations in a single process.
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
        dfcli_module="phyloframe.legacy._alifestd_unminify_asexual",
        dfcli_version=get_phyloframe_version(),
    )
    return parser


if __name__ == "__main__":
    begin_prod_logging()

    parser = _create_parser()
    args, __ = parser.parse_known_args()
    with log_context_duration(
        "phyloframe.legacy._alifestd_unminify_asexual", logging.info
    ):
        _run_dataframe_cli(
            base_parser=parser,
            output_dataframe_op=delegate_polars_implementation()(
                alifestd_unminify_asexual,
            ),
        )
