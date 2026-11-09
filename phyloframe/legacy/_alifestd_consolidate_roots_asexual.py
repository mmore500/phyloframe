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
from ._alifestd_find_oldest_root_id_asexual import (
    alifestd_find_oldest_root_id_asexual,
)
from ._alifestd_is_asexual import alifestd_is_asexual
from ._alifestd_make_ancestor_list_col import alifestd_make_ancestor_list_col
from ._alifestd_topological_sensitivity_warned import (
    alifestd_topological_sensitivity_warned,
)
from ._alifestd_try_add_ancestor_id_col import alifestd_try_add_ancestor_id_col


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
    """Delete all roots but the oldest, pointing their children to it.

    Unlike `alifestd_join_roots`, which points the other roots themselves
    to the oldest root, removes them; roots without children are removed
    altogether. Ids are not reassigned, and rows may no longer be
    topologically sorted.

    Parameters
    ----------
    phylogeny_df : pd.DataFrame
        Phylogeny in alife standard format.

        Must represent an asexual phylogeny.
    mutate : bool, default False
        If True, allows in-place modification of `phylogeny_df`.
    root_ancestor_token : str, default "none"
        Token sandwiched in brackets as the root's `ancestor_list` entry, if
        that column is present; e.g., "none" gives "[none]".

    Returns
    -------
    pd.DataFrame
        The phylogeny with one root, the oldest (see
        `alifestd_find_oldest_root_id_asexual`).

    Raises
    ------
    ValueError
        If `phylogeny_df` is not asexual.

    Warns
    -----
    UserWarning
        If a root's `origin_time` is null or nan; such roots sort last in
        finding the oldest.

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

    oldest_root_id = alifestd_find_oldest_root_id_asexual(
        phylogeny_df, mutate=True
    )
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
