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
from ._alifestd_is_asexual import alifestd_is_asexual
from ._alifestd_make_ancestor_list_col import alifestd_make_ancestor_list_col
from ._alifestd_mark_oldest_root import alifestd_mark_oldest_root
from ._alifestd_topological_sensitivity_warned import (
    alifestd_topological_sensitivity_warned,
)
from ._alifestd_try_add_ancestor_id_col import alifestd_try_add_ancestor_id_col


def _find_oldest_root_id(phylogeny_df: pd.DataFrame) -> int:
    """Implementation detail for `alifestd_consolidate_roots_asexual`.

    Returns the id of the oldest root, as for `alifestd_mark_oldest_root`,
    raising NotImplementedError if a root's `origin_time` is null or nan, as
    then which is oldest is undefined.
    """
    columns = [
        column
        for column in ("id", "ancestor_id", "origin_time")
        if column in phylogeny_df
    ]
    roots_df = phylogeny_df.loc[
        phylogeny_df["id"] == phylogeny_df["ancestor_id"], columns
    ]
    if "origin_time" in roots_df and roots_df["origin_time"].isna().any():
        raise NotImplementedError(
            "null or nan root origin_time not supported, as then which root "
            "is oldest is undefined",
        )
    marked_df = alifestd_mark_oldest_root(roots_df, mark_as="is_oldest_root")
    return marked_df.loc[marked_df["is_oldest_root"], "id"].item()


@alifestd_topological_sensitivity_warned(
    insert=False,
    delete=True,
    update=True,
)
def alifestd_consolidate_roots_asexual(
    phylogeny_df: pd.DataFrame,
    mutate: bool = False,
    root_ancestor_token: str = "none",
) -> pd.DataFrame:
    """Delete all roots but the oldest, measured by lowest `origin_time` (if
    available) or otherwise lowest `id`, pointing their children to the
    oldest root.

    Unlike `alifestd_join_roots`, which points the other roots themselves to
    the oldest root, removes them, so that their children become children of
    the oldest root. Roots without children are removed altogether. So, the
    result is one tree, and no new branches start from the other roots'
    origin times, e.g., where those are placeholders.

    Ids are not reassigned, so may become non-contiguous, and rows may no
    longer be topologically sorted if the oldest root follows the other
    roots' children. Dataframe reindexing (e.g., df.index) may be applied.

    The option `root_ancestor_token` will be sandwiched in brackets to create
    the ancestor list entry for genesis organisms, if an `ancestor_list`
    column is present. For example, the token "None" will yield the entry
    "[None]" and the token "" will yield the entry "[]". Default "none".

    Raises ValueError if `phylogeny_df` is not asexual, and
    NotImplementedError if a root's `origin_time` is null or nan.

    Input dataframe is not mutated by this operation unless `mutate` set True.
    If mutate set True, operation does not occur in place; still use return
    value to get transformed phylogeny dataframe.

    See Also
    --------
    alifestd_consolidate_roots_polars :
        Polars-based implementation.
    alifestd_join_roots :
        Points the other roots to the oldest root, keeping them.
    alifestd_delete_unifurcating_roots_asexual :
        Deletes roots with one child, which then become roots.
    """
    if not mutate:
        phylogeny_df = phylogeny_df.copy()

    if not alifestd_is_asexual(phylogeny_df):
        raise ValueError("phylogeny_df must be strictly asexual")

    phylogeny_df = alifestd_try_add_ancestor_id_col(phylogeny_df, mutate=True)
    if phylogeny_df.empty:
        return phylogeny_df

    oldest_root_id = _find_oldest_root_id(phylogeny_df)
    is_root = phylogeny_df["id"] == phylogeny_df["ancestor_id"]
    is_other_root = is_root & (phylogeny_df["id"] != oldest_root_id)
    other_root_ids = phylogeny_df.loc[is_other_root, "id"]

    phylogeny_df.loc[
        ~is_root & phylogeny_df["ancestor_id"].isin(other_root_ids),
        "ancestor_id",
    ] = oldest_root_id
    phylogeny_df = phylogeny_df.loc[~is_other_root].reset_index(drop=True)

    if "ancestor_list" in phylogeny_df:
        phylogeny_df["ancestor_list"] = alifestd_make_ancestor_list_col(
            phylogeny_df["id"],
            phylogeny_df["ancestor_id"],
            root_ancestor_token=root_ancestor_token,
        )

    return phylogeny_df


_raw_description = f"""{os.path.basename(__file__)} | (phyloframe v{get_phyloframe_version()}/joinem v{joinem.__version__})

Delete all roots but the oldest, measured by lowest `origin_time` (if available) or otherwise lowest `id`, pointing their children to the oldest root.

Roots without children are removed altogether.

Data is assumed to be in alife standard format.
Only supports asexual phylogenies.

Additional Notes
================
- Ids are not reassigned, and topological sort may not be preserved.

- Use `--eager-read` if modifying data file inplace.

- This CLI entrypoint is experimental and may be subject to change.

See Also
========
phyloframe.legacy._alifestd_consolidate_roots_polars :
    CLI entrypoint for Polars-based implementation.
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
        dfcli_module="phyloframe.legacy._alifestd_consolidate_roots_asexual",
        dfcli_version=get_phyloframe_version(),
    )
    parser.add_argument(
        "--root-ancestor-token",
        type=str,
        default="none",
        help='token for root ancestor entries (default: "none")',
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
        "phyloframe.legacy._alifestd_consolidate_roots_asexual",
        logging.info,
    ):
        _run_dataframe_cli(
            base_parser=parser,
            output_dataframe_op=delegate_polars_implementation()(
                functools.partial(
                    alifestd_consolidate_roots_asexual,
                    root_ancestor_token=args.root_ancestor_token,
                    ignore_topological_sensitivity=args.ignore_topological_sensitivity,
                    drop_topological_sensitivity=args.drop_topological_sensitivity,
                ),
            ),
        )
