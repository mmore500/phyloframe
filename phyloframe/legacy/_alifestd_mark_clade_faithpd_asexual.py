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
from ._alifestd_mark_origin_time_delta_asexual import (
    alifestd_mark_origin_time_delta_asexual,
)
from ._alifestd_topological_sort import alifestd_topological_sort
from ._alifestd_try_add_ancestor_id_col import alifestd_try_add_ancestor_id_col


@jit(nopython=True)
def _alifestd_mark_clade_faithpd_asexual_fast_path(
    ancestor_ids: np.ndarray,
    origin_time_deltas: np.ndarray,
) -> np.ndarray:
    """Implementation detail for `alifestd_mark_clade_faithpd_asexual`."""

    clade_faithpds = np.zeros_like(origin_time_deltas)
    for idx_r, ancestor_id in enumerate(ancestor_ids[::-1]):
        idx = len(ancestor_ids) - 1 - idx_r
        if ancestor_id == idx:
            continue  # handle root cases

        clade_faithpds[ancestor_id] += (
            origin_time_deltas[idx] + clade_faithpds[idx]
        )

    return clade_faithpds


def _alifestd_mark_clade_faithpd_asexual_slow_path(
    phylogeny_df: pd.DataFrame,
) -> pd.DataFrame:
    """Implementation detail for `alifestd_mark_clade_faithpd_asexual`."""
    phylogeny_df.index = phylogeny_df["id"]

    phylogeny_df["clade_faithpd"] = phylogeny_df["origin_time_delta"].copy()
    phylogeny_df["clade_faithpd"] = 0

    for idx in reversed(phylogeny_df.index):
        ancestor_id = phylogeny_df.at[idx, "ancestor_id"]
        if ancestor_id == idx:
            continue  # handle root cases

        phylogeny_df.at[ancestor_id, "clade_faithpd"] += (
            phylogeny_df.at[idx, "origin_time_delta"]
            + phylogeny_df.at[idx, "clade_faithpd"]
        )

    return phylogeny_df


def alifestd_mark_clade_faithpd_asexual(
    phylogeny_df: pd.DataFrame,
    mutate: bool = False,
    mark_as: str = "clade_faithpd",
) -> pd.DataFrame:
    """Add column `clade_faithpd`, containing sum branch length among
    descendant noes.

    The output column name can be changed via the ``mark_as`` parameter.

    Branch length is defined as the difference between the origin time
    of the node and the origin time of its ancestor.

    A topological sort will be applied if `phylogeny_df` is not topologically
    sorted. Dataframe reindexing (e.g., df.index) may be applied.

    Input dataframe is not mutated by this operation unless `mutate` set True.
    If mutate set True, operation does not occur in place; still use return
    value to get transformed phylogeny dataframe.
    """

    if not mutate:
        phylogeny_df = phylogeny_df.copy()

    phylogeny_df = alifestd_try_add_ancestor_id_col(phylogeny_df, mutate=True)

    if not alifestd_is_topologically_sorted(phylogeny_df):
        phylogeny_df = alifestd_topological_sort(phylogeny_df, mutate=True)

    if "origin_time_delta" not in phylogeny_df.columns:
        phylogeny_df = alifestd_mark_origin_time_delta_asexual(
            phylogeny_df, mutate=True
        )

    if (
        pd.api.types.is_integer_dtype(phylogeny_df["origin_time_delta"].dtype)
        and phylogeny_df["origin_time_delta"].dtype != np.uint64
    ):
        phylogeny_df["origin_time_delta_"] = phylogeny_df[
            "origin_time_delta"
        ].astype(np.int64)

    if alifestd_has_contiguous_ids(phylogeny_df):
        phylogeny_df[mark_as] = _alifestd_mark_clade_faithpd_asexual_fast_path(
            pd.to_numeric(phylogeny_df["ancestor_id"]).to_numpy(),
            pd.to_numeric(phylogeny_df["origin_time_delta"]).to_numpy(),
        )
        return phylogeny_df
    else:
        phylogeny_df = _alifestd_mark_clade_faithpd_asexual_slow_path(
            phylogeny_df,
        )
        if mark_as != "clade_faithpd":
            phylogeny_df.rename(
                columns={"clade_faithpd": mark_as},
                inplace=True,
            )
        return phylogeny_df


_raw_description = f"""{os.path.basename(__file__)} | (phyloframe v{get_phyloframe_version()}/joinem v{joinem.__version__})

Add column `clade_faithpd`, containing sum branch length among descendant nodes.

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
        dfcli_module="phyloframe.legacy._alifestd_mark_clade_faithpd_asexual",
        dfcli_version=get_phyloframe_version(),
    )
    parser.add_argument(
        "--mark-as",
        default="clade_faithpd",
        type=str,
        help="output column name (default: clade_faithpd)",
    )
    return parser


if __name__ == "__main__":
    begin_prod_logging()

    parser = _create_parser()
    args, __ = parser.parse_known_args()
    with log_context_duration(
        "phyloframe.legacy._alifestd_mark_clade_faithpd_asexual",
        logging.info,
    ):
        _run_dataframe_cli(
            base_parser=parser,
            output_dataframe_op=delegate_polars_implementation()(
                functools.partial(
                    alifestd_mark_clade_faithpd_asexual, mark_as=args.mark_as
                ),
            ),
        )
