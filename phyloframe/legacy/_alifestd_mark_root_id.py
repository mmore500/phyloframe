import argparse
import functools
import logging
import os
import typing

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
from ._alifestd_parse_ancestor_ids import alifestd_parse_ancestor_ids
from ._alifestd_topological_sort import alifestd_topological_sort
from ._alifestd_try_add_ancestor_id_col import alifestd_try_add_ancestor_id_col


@jit(nopython=True)
def _alifestd_mark_root_id_asexual_fast_path(
    ancestor_ids: np.ndarray,
) -> np.ndarray:
    """Implementation detail for `alifestd_mark_root_id`."""
    root_ids = np.arange(len(ancestor_ids), dtype=ancestor_ids.dtype)
    for idx in range(len(ancestor_ids)):
        ancestor_id = ancestor_ids[idx]
        root_ids[idx] = root_ids[ancestor_id]
    return root_ids


def alifestd_mark_root_id(
    phylogeny_df: pd.DataFrame,
    mutate: bool = False,
    selector: typing.Callable = min,
    *,
    mark_as: str = "root_id",
) -> pd.DataFrame:
    """Add column `root_id`, containing the `id` of entries' ultimate ancestor.

    The output column name can be changed via the ``mark_as`` parameter.

    For sexual data, the field `root_id` is chosen according to the selection
    of callable `selector` over parents' `root_id` values. Note that subsets
    within a connected component may be marked with different `root_id` values.
    To create a component id that is consistent within connected components,
    a backward pass could be performed that updates ancestors' values if they
    are greater than that of each descendant.

    Input dataframe is not mutated by this operation unless `mutate` set True.
    If mutate set True, operation does not occur in place; still use return
    value to get transformed phylogeny dataframe.
    """

    if not mutate:
        phylogeny_df = phylogeny_df.copy()

    phylogeny_df = alifestd_try_add_ancestor_id_col(phylogeny_df, mutate=True)

    if not alifestd_is_topologically_sorted(phylogeny_df):
        phylogeny_df = alifestd_topological_sort(phylogeny_df, mutate=True)

    if (
        alifestd_has_contiguous_ids(phylogeny_df)
        and "ancestor_id" in phylogeny_df.columns
    ):
        phylogeny_df[mark_as] = _alifestd_mark_root_id_asexual_fast_path(
            phylogeny_df["ancestor_id"].to_numpy(),
        )
    elif "ancestor_id" in phylogeny_df.columns:  # asexual, non-contiguous
        phylogeny_df.index = phylogeny_df["id"]
        phylogeny_df[mark_as] = phylogeny_df["id"]
        for index in phylogeny_df.index:
            ancestor_id = phylogeny_df.at[index, "ancestor_id"]
            phylogeny_df.at[index, mark_as] = phylogeny_df.at[
                ancestor_id, mark_as
            ]
    else:  # sexual
        phylogeny_df.index = phylogeny_df["id"]
        phylogeny_df[mark_as] = phylogeny_df["id"]
        for index in phylogeny_df.index:
            ancestor_list = phylogeny_df.at[index, "ancestor_list"]
            ancestor_ids = alifestd_parse_ancestor_ids(ancestor_list)
            candidate_roots = [
                phylogeny_df.at[aid, mark_as] for aid in ancestor_ids
            ]
            # "or" covers genesis empty list case
            phylogeny_df.at[index, mark_as] = selector(
                candidate_roots or [index]
            )

    return phylogeny_df


_raw_description = f"""{os.path.basename(__file__)} | (phyloframe v{get_phyloframe_version()}/joinem v{joinem.__version__})

Add column `root_id`, containing the `id` of entries' ultimate ancestor.

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
        dfcli_module="phyloframe.legacy._alifestd_mark_root_id",
        dfcli_version=get_phyloframe_version(),
    )
    parser.add_argument(
        "--mark-as",
        default="root_id",
        type=str,
        help="output column name (default: root_id)",
    )
    return parser


if __name__ == "__main__":
    begin_prod_logging()

    parser = _create_parser()
    args, __ = parser.parse_known_args()
    with log_context_duration(
        "phyloframe.legacy._alifestd_mark_root_id", logging.info
    ):
        _run_dataframe_cli(
            base_parser=parser,
            output_dataframe_op=delegate_polars_implementation()(
                functools.partial(alifestd_mark_root_id, mark_as=args.mark_as),
            ),
        )
