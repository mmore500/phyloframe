import argparse
import functools
import logging
import os
import warnings

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
from ._alifestd_to_working_format import alifestd_to_working_format
from ._alifestd_unfurl_traversal_preorder_asexual import (
    alifestd_unfurl_traversal_preorder_asexual,
)


@jit(nopython=True)
def _alifestd_calc_minified_ancestor_offsets_asexual(
    ancestor_ids: np.ndarray,
    order: np.ndarray,
) -> np.ndarray:
    """Compute `ancestor_offset` values for rows reordered by `order`.

    Parameters
    ----------
    ancestor_ids : np.ndarray
        Array of ancestor IDs, assumed contiguous (ids == row indices).
    order : np.ndarray
        Row permutation where each node appears after its ancestor (e.g.,
        a DFS preorder traversal).

    Returns
    -------
    np.ndarray
        Array where entry i is the number of rows between reordered row i
        and its ancestor's reordered row (zero for roots).
    """
    n = len(order)
    dtype = ancestor_ids.dtype
    position = np.empty(n, dtype=dtype)
    for pos in range(n):
        position[order[pos]] = pos

    result = np.empty(n, dtype=dtype)
    for pos in range(n):
        pos_ = dtype.type(pos)  # prevent numba float64 promotion
        result[pos] = pos_ - position[ancestor_ids[order[pos]]]

    return result


def alifestd_minify_asexual(
    phylogeny_df: pd.DataFrame,
    shrink_dtype: bool = True,
    mutate: bool = False,
) -> pd.DataFrame:
    """Re-encode phylogeny_df compactly, replacing explicit ids with
    relative ancestor row offsets.

    The returned phylogeny dataframe will
    * be sorted in DFS preorder (i.e., organisms appear after their
    ancestor, with subtrees in contiguous rows, so that offspring tend to
    be close to their parents),
    * have no `id`, `ancestor_id`, or `ancestor_list` columns,
    * contain an `ancestor_offset` column, giving the number of rows back to
    each organism's ancestor (zero for roots).

    Each organism's ancestor is at row position `pos - ancestor_offset`.
    Use `alifestd_unminify_asexual` to restore alife standard format.

    If `shrink_dtype` is set True (the default), `ancestor_offset` will use
    the smallest signed integer dtype that can hold all offsets. Otherwise,
    it will have the same dtype as `ancestor_id`.

    The provided dataframe must be asexual. Any existing `ancestor_offset`
    column will be overwritten. Dataframe reindexing (e.g., df.index) may be
    applied.

    Input dataframe is not mutated by this operation unless `mutate` set True.
    If mutate set True, operation does not occur in place; still use return
    value to get transformed phylogeny dataframe.

    See Also
    --------
    alifestd_minify_polars :
        Polars-based implementation.
    alifestd_unminify_asexual :
        Inverse operation, restoring alife standard format.
    """
    if "ancestor_offset" in phylogeny_df.columns:
        warnings.warn(
            "alifestd_minify_asexual overwrites existing ancestor_offset",
        )

    phylogeny_df = alifestd_to_working_format(phylogeny_df, mutate=mutate)

    order = alifestd_unfurl_traversal_preorder_asexual(
        phylogeny_df, mutate=True
    )
    ancestor_offsets = _alifestd_calc_minified_ancestor_offsets_asexual(
        phylogeny_df["ancestor_id"].to_numpy(),
        order,
    )

    if shrink_dtype:
        # cast to int before negating, in case offsets are unsigned
        offset_dtype = np.min_scalar_type(
            -int(ancestor_offsets.max(initial=0) or 1),
        )
        logging.info(
            "- alifestd_minify_asexual: "
            f"shrinking ancestor_offset dtype to {offset_dtype}...",
        )
        ancestor_offsets = ancestor_offsets.astype(offset_dtype)

    phylogeny_df = phylogeny_df.iloc[order].assign(
        ancestor_offset=ancestor_offsets,
    )
    return phylogeny_df.drop(
        columns=["id", "ancestor_id", "ancestor_list"],
        errors="ignore",
    )


_raw_description = f"""{os.path.basename(__file__)} | (phyloframe v{get_phyloframe_version()}/joinem v{joinem.__version__})

Re-encode phylogeny compactly, replacing explicit ids with relative ancestor row offsets.

The output phylogeny will
- be sorted in DFS preorder (i.e., organisms appear after their ancestor, with subtrees in contiguous rows, so that offspring tend to be close to their parents),
- have no `id`, `ancestor_id`, or `ancestor_list` columns,
- contain an `ancestor_offset` column, giving the number of rows back to each organism's ancestor (zero for roots).

By default, `ancestor_offset` uses the smallest signed integer dtype that can hold all offsets.
Use `--no-shrink-dtype` to instead keep the `ancestor_id` dtype.

Data is assumed to be in alife standard format.

Additional Notes
================
- Use `--eager-read` if modifying data file inplace.

- This CLI entrypoint is experimental and may be subject to change.

See Also
========
phyloframe.legacy._alifestd_minify_polars :
    Entrypoint for high-performance Polars-based implementation.
phyloframe.legacy._alifestd_unminify_asexual :
    CLI entrypoint for inverse operation, restoring alife standard format.
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
        dfcli_module="phyloframe.legacy._alifestd_minify_asexual",
        dfcli_version=get_phyloframe_version(),
    )
    add_bool_arg(
        parser,
        "shrink-dtype",
        default=True,
        help=(
            "use the smallest integer dtype that can hold `ancestor_offset` "
            "values (default: True)"
        ),
    )
    return parser


if __name__ == "__main__":
    begin_prod_logging()

    parser = _create_parser()
    args, __ = parser.parse_known_args()
    with log_context_duration(
        "phyloframe.legacy._alifestd_minify_asexual", logging.info
    ):
        _run_dataframe_cli(
            base_parser=parser,
            output_dataframe_op=delegate_polars_implementation()(
                functools.partial(
                    alifestd_minify_asexual,
                    shrink_dtype=args.shrink_dtype,
                ),
            ),
        )
