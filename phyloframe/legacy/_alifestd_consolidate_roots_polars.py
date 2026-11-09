import argparse
import functools
import logging
import os

import joinem
from joinem._dataframe_cli import _add_parser_base, _run_dataframe_cli
import polars as pl

from .._auxlib._add_bool_arg import add_bool_arg
from .._auxlib._begin_prod_logging import begin_prod_logging
from .._auxlib._format_cli_description import format_cli_description
from .._auxlib._get_phyloframe_version import get_phyloframe_version
from .._auxlib._log_context_duration import log_context_duration
from ._alifestd_make_ancestor_list_col_polars import (
    alifestd_make_ancestor_list_col_polars,
)
from ._alifestd_mark_oldest_root_polars import (
    alifestd_mark_oldest_root_polars,
)
from ._alifestd_topological_sensitivity_warned_polars import (
    alifestd_topological_sensitivity_warned_polars,
)
from ._alifestd_try_add_ancestor_id_col_polars import (
    alifestd_try_add_ancestor_id_col_polars,
)


def _find_oldest_root_id(phylogeny_df: pl.DataFrame) -> int:
    """Implementation detail for `alifestd_consolidate_roots_polars`.

    Returns the id of the oldest root, as for
    `alifestd_mark_oldest_root_polars`, raising NotImplementedError if a
    root's `origin_time` is null or nan, as then which is oldest is
    undefined.
    """
    schema_names = phylogeny_df.lazy().collect_schema().names()
    roots_df = (
        phylogeny_df.lazy()
        .select(
            "id",
            "ancestor_id",
            *(["origin_time"] if "origin_time" in schema_names else []),
        )
        .filter(pl.col("id") == pl.col("ancestor_id"))
        .collect()
    )
    if "origin_time" in schema_names:
        origin_time = roots_df["origin_time"]
        if origin_time.has_nulls() or (
            origin_time.dtype.is_float() and origin_time.is_nan().any()
        ):
            raise NotImplementedError(
                "null or nan root origin_time not supported, as then which "
                "root is oldest is undefined",
            )
    return (
        alifestd_mark_oldest_root_polars(roots_df, mark_as="is_oldest_root")
        .filter(pl.col("is_oldest_root"))["id"]
        .item()
    )


def _update_ancestor_list(
    phylogeny_df: pl.DataFrame, *, root_ancestor_token: str
) -> pl.DataFrame:
    """Implementation detail for `alifestd_consolidate_roots_polars`.

    Returns `phylogeny_df`, of the same type, with `ancestor_list` remade
    from `ancestor_id` (see `alifestd_make_ancestor_list_col_polars`).
    """
    collected = phylogeny_df.lazy().collect()
    updated = collected.with_columns(
        ancestor_list=alifestd_make_ancestor_list_col_polars(
            collected["id"],
            collected["ancestor_id"],
            root_ancestor_token=root_ancestor_token,
        ),
    )
    return (
        updated.lazy() if isinstance(phylogeny_df, pl.LazyFrame) else updated
    )


@alifestd_topological_sensitivity_warned_polars(
    insert=False,
    delete=True,
    update=True,
)
def alifestd_consolidate_roots_polars(
    phylogeny_df: pl.DataFrame,
    *,
    root_ancestor_token: str = "none",
) -> pl.DataFrame:
    """Delete all roots but the oldest, measured by lowest `origin_time` (if
    available) or otherwise lowest `id`, pointing their children to the
    oldest root.

    Unlike `alifestd_join_roots_polars`, which points the other roots
    themselves to the oldest root, removes them, so that their children
    become children of the oldest root. Roots without children are removed
    altogether. So, the result is one tree, and no new branches start from
    the other roots' origin times, e.g., where those are placeholders.

    Ids are not reassigned, so may become non-contiguous (see
    `alifestd_assign_contiguous_ids_polars`), and rows may no longer be
    topologically sorted if the oldest root follows the other roots'
    children (see `alifestd_topological_sort_polars`).

    Parameters
    ----------
    phylogeny_df : polars.DataFrame or polars.LazyFrame
        The phylogeny as a dataframe in alife standard format.

        Must represent an asexual phylogeny. If there is no `ancestor_id`
        column, it is added from `ancestor_list`; an `ancestor_list` column,
        if present, is updated.
    root_ancestor_token : str, default "none"
        Token sandwiched in brackets as the `ancestor_list` entry of the
        root, if `ancestor_list` is present; e.g., "none" gives "[none]".

    Returns
    -------
    polars.DataFrame or polars.LazyFrame
        The phylogeny with one root, of the same type as `phylogeny_df`.

    Raises
    ------
    NotImplementedError
        If `phylogeny_df` has no `ancestor_id` column and none can be added
        from `ancestor_list`, or if a root's `origin_time` is null or nan.

    See Also
    --------
    alifestd_consolidate_roots_asexual :
        Pandas-based implementation.
    alifestd_join_roots_polars :
        Points the other roots to the oldest root, keeping them.
    alifestd_delete_unifurcating_roots_polars :
        Deletes roots with one child, which then become roots.
    """
    if "ancestor_list" in phylogeny_df.lazy().collect_schema().names():
        phylogeny_df = alifestd_try_add_ancestor_id_col_polars(phylogeny_df)
    schema = phylogeny_df.lazy().collect_schema()
    if "ancestor_id" not in schema.names():
        raise NotImplementedError(
            "ancestor_id or ancestor_list column required",
        )

    if phylogeny_df.lazy().limit(1).collect().is_empty():
        return phylogeny_df

    logging.info(
        "- alifestd_consolidate_roots_polars: finding oldest root...",
    )
    oldest_root_id = _find_oldest_root_id(phylogeny_df)

    is_root = pl.col("id") == pl.col("ancestor_id")
    other_root_ids = (
        phylogeny_df.lazy()
        .filter(is_root & (pl.col("id") != oldest_root_id))
        .select("id")
        .collect()["id"]
    )
    logging.info(
        f"- alifestd_consolidate_roots_polars: deleting {len(other_root_ids)} "
        "other roots...",
    )
    consolidated = phylogeny_df.with_columns(
        ancestor_id=pl.when(
            ~is_root & pl.col("ancestor_id").is_in(other_root_ids.implode()),
        )
        .then(pl.lit(oldest_root_id, dtype=schema["ancestor_id"]))
        .otherwise(pl.col("ancestor_id")),
    ).filter(~pl.col("id").is_in(other_root_ids.implode()))
    if "ancestor_list" not in schema.names():
        return consolidated
    return _update_ancestor_list(
        consolidated, root_ancestor_token=root_ancestor_token
    )


_raw_description = f"""{os.path.basename(__file__)} | (phyloframe v{get_phyloframe_version()}/joinem v{joinem.__version__})

Delete all roots but the oldest, measured by lowest `origin_time` (if available) or otherwise lowest `id`, pointing their children to the oldest root.

Roots without children are removed altogether.

Data is assumed to be in alife standard format.
Only supports asexual phylogenies.

Additional Notes
================
- 'ancestor_id' is added from 'ancestor_list' if missing; 'ancestor_list', if
  present, is updated.

- Ids are not reassigned, and topological sort may not be preserved.

- Use `--eager-read` if modifying data file inplace.

- This CLI entrypoint is experimental and may be subject to change.

See Also
========
phyloframe.legacy._alifestd_consolidate_roots_asexual :
    CLI entrypoint for Pandas-based implementation.
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
        dfcli_module="phyloframe.legacy._alifestd_consolidate_roots_polars",
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

    try:
        with log_context_duration(
            "phyloframe.legacy._alifestd_consolidate_roots_polars",
            logging.info,
        ):
            _run_dataframe_cli(
                base_parser=parser,
                output_dataframe_op=functools.partial(
                    alifestd_consolidate_roots_polars,
                    root_ancestor_token=args.root_ancestor_token,
                    ignore_topological_sensitivity=args.ignore_topological_sensitivity,
                    drop_topological_sensitivity=args.drop_topological_sensitivity,
                ),
            )
    except NotImplementedError as e:
        logging.error(
            "- polars op not yet implemented, use pandas op CLI instead",
        )
        raise e
