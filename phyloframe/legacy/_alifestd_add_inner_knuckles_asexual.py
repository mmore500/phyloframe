import argparse
import functools
import logging
import os

import joinem
from joinem._dataframe_cli import _add_parser_base, _run_dataframe_cli
import pandas as pd

from .._auxlib._add_bool_arg import add_bool_arg
from .._auxlib._begin_prod_logging import begin_prod_logging
from .._auxlib._delegate_polars_implementation import (
    delegate_polars_implementation,
)
from .._auxlib._format_cli_description import format_cli_description
from .._auxlib._get_phyloframe_version import get_phyloframe_version
from .._auxlib._log_context_duration import log_context_duration
from .._auxlib._preserve_id_dtypes import preserve_id_dtypes
from ._alifestd_make_ancestor_list_col import alifestd_make_ancestor_list_col
from ._alifestd_mark_leaves import alifestd_mark_leaves
from ._alifestd_topological_sensitivity_warned import (
    alifestd_topological_sensitivity_warned,
)
from ._alifestd_try_add_ancestor_id_col import alifestd_try_add_ancestor_id_col


@preserve_id_dtypes
@alifestd_topological_sensitivity_warned(
    insert=True,
    delete=False,
    update=True,
)
def alifestd_add_inner_knuckles_asexual(
    phylogeny_df: pd.DataFrame, mutate: bool = False
) -> pd.DataFrame:
    """For all inner nodes, add a subtending unifurcation ("knuckle").

    Input dataframe is not mutated by this operation unless `mutate` set True.
    If mutate set True, operation does not occur in place; still use return
    value to get transformed phylogeny dataframe.
    """
    if not mutate:
        phylogeny_df = phylogeny_df.copy()

    phylogeny_df = alifestd_mark_leaves(phylogeny_df, mutate=True)

    phylogeny_df = alifestd_try_add_ancestor_id_col(phylogeny_df, mutate=True)
    if "ancestor_id" not in phylogeny_df:
        raise ValueError("asexual phylogeny required")

    inner_df = phylogeny_df[~phylogeny_df["is_leaf"]].copy()

    if inner_df.empty:
        return phylogeny_df

    if "is_root" in inner_df:
        inner_df["is_root"] = False

    if "origin_time_delta" in inner_df:
        inner_df["origin_time_delta"] = 0

    id_delta = phylogeny_df["id"].max() + 1

    inner_df["ancestor_id"] += id_delta * (
        inner_df["id"] == inner_df["ancestor_id"]
    )
    inner_df["id"] += id_delta

    if not (inner_df["id"].min() > phylogeny_df["id"].max()):
        raise ValueError("overflow in new id assigment")

    phylogeny_df.loc[~phylogeny_df["is_leaf"], "ancestor_id"] = (
        phylogeny_df.loc[~phylogeny_df["is_leaf"], "id"] + id_delta
    )

    res = pd.concat([phylogeny_df, inner_df], ignore_index=True)

    if "ancestor_list" in res:
        res["ancestor_list"] = alifestd_make_ancestor_list_col(
            res["id"],
            res["ancestor_id"],
        )

    return res


_raw_description = f"""{os.path.basename(__file__)} | (phyloframe v{get_phyloframe_version()}/joinem v{joinem.__version__})

For all inner nodes, add a subtending unifurcation (knuckle).

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
        dfcli_module="phyloframe.legacy._alifestd_add_inner_knuckles_asexual",
        dfcli_version=get_phyloframe_version(),
    )
    add_bool_arg(
        parser,
        "ignore-topological-sensitivity",
        default=False,
        help="suppress topological sensitivity warning (default: False)",
    )
    add_bool_arg(
        parser,
        "drop-topological-sensitivity",
        default=False,
        help="drop topology-sensitive columns from output (default: False)",
    )
    return parser


if __name__ == "__main__":
    begin_prod_logging()

    parser = _create_parser()
    args, __ = parser.parse_known_args()
    with log_context_duration(
        "phyloframe.legacy._alifestd_add_inner_knuckles_asexual",
        logging.info,
    ):
        _run_dataframe_cli(
            base_parser=parser,
            output_dataframe_op=delegate_polars_implementation()(
                functools.partial(
                    alifestd_add_inner_knuckles_asexual,
                    ignore_topological_sensitivity=args.ignore_topological_sensitivity,
                    drop_topological_sensitivity=args.drop_topological_sensitivity,
                ),
            ),
        )
