import argparse
import logging
import numbers
import os
import typing
import warnings

import joinem
from joinem._dataframe_cli import _add_parser_base, _run_dataframe_cli
import opytional as opyt
import polars as pl

from .._auxlib._add_bool_arg import add_bool_arg
from .._auxlib._begin_prod_logging import begin_prod_logging
from .._auxlib._format_cli_description import format_cli_description
from .._auxlib._get_phyloframe_version import get_phyloframe_version
from .._auxlib._log_context_duration import log_context_duration
from ._alifestd_topological_sensitivity_warned_polars import (
    alifestd_topological_sensitivity_warned_polars,
)


@alifestd_topological_sensitivity_warned_polars(
    insert=True,
    delete=False,
    update=True,
)
def alifestd_prefix_roots_polars(
    phylogeny_df: pl.DataFrame,
    *,
    allow_id_reassign: bool = False,
    origin_time: typing.Optional[numbers.Real] = None,
) -> pl.DataFrame:
    """Add new roots to the phylogeny, prefixing existing roots.

    An origin time may be specified, in which case only roots with origin times
    past the specified time will be prefixed. If no origin time is specified,
    all roots will be prefixed.
    """
    if "origin_time_delta" in phylogeny_df:
        warnings.warn("alifestd_prefix_roots ignores origin_time_delta values")
    if origin_time is not None and "origin_time" not in phylogeny_df:
        raise ValueError(
            "origin_time specified but not present in phylogeny dataframe",
        )

    if "ancestor_list" in phylogeny_df:
        raise NotImplementedError
    if not allow_id_reassign:
        raise NotImplementedError
    if phylogeny_df.lazy().limit(1).collect().is_empty():
        raise NotImplementedError
    has_contiguous_ids = phylogeny_df.select(
        pl.col("id").diff() == 1
    ).to_series().all() and (phylogeny_df["id"].first() == 0)
    if not has_contiguous_ids:
        raise NotImplementedError

    phylogeny_df = phylogeny_df.drop("is_root", strict=False)

    logging.info("- alifestd_prefix_roots: identifying eligible roots...")
    is_root_expr = pl.col("id") == pl.col("ancestor_id")
    is_eligible_expr = (
        (pl.col("origin_time") > origin_time) & is_root_expr
        if origin_time is not None
        else is_root_expr
    )

    logging.info("- alifestd_prefix_roots: filtering prepended roots...")
    prefix_roots = (
        phylogeny_df.lazy()
        .filter(is_eligible_expr)
        .select("id", "ancestor_id", "origin_time")
        .collect()
    )
    num_prepended = len(prefix_roots)
    if num_prepended == 0:
        return phylogeny_df

    logging.info("- alifestd_prefix_roots: building ancestor remap table...")
    root_original_ids = prefix_roots["id"]
    new_ancestor_map = pl.DataFrame(
        {
            "row_idx": root_original_ids,
            "new_ancestor_id": pl.int_range(
                num_prepended,
                eager=True,
            ).cast(root_original_ids.dtype),
        },
    )

    logging.info("- alifestd_prefix_roots: shifting ids and remapping...")
    phylogeny_df = (
        phylogeny_df.lazy()
        .with_columns(
            id=pl.col("id") + num_prepended,
            ancestor_id=pl.col("ancestor_id") + num_prepended,
        )
        .with_row_index("row_idx")
        .join(new_ancestor_map.lazy(), on="row_idx", how="left")
        .with_columns(
            ancestor_id=pl.coalesce("new_ancestor_id", "ancestor_id"),
        )
        .drop("row_idx", "new_ancestor_id")
        .collect()
    )

    logging.info("- alifestd_prefix_roots: building prepended root rows...")
    prefix_roots = prefix_roots.with_columns(
        id=pl.int_range(num_prepended),
        ancestor_id=pl.int_range(num_prepended),
        origin_time=pl.lit(opyt.or_value(origin_time, 0)),
    ).cast(
        {
            k: v
            for k, v in phylogeny_df.collect_schema().items()
            if k in prefix_roots.collect_schema()
        },
    )

    prefix_roots = prefix_roots.with_columns(
        **{
            col: pl.lit(None).cast(phylogeny_df[col].dtype)
            for col in set(phylogeny_df.columns) - set(prefix_roots.columns)
        },
    ).select(phylogeny_df.columns)

    logging.info("- alifestd_prefix_roots: concatenating result...")
    return pl.concat([prefix_roots, phylogeny_df])


_raw_description = f"""{os.path.basename(__file__)} | (phyloframe v{get_phyloframe_version()}/joinem v{joinem.__version__})

Add new roots to the phylogeny, prefixing existing roots.

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
        dfcli_module="phyloframe.legacy._alifestd_prefix_roots_polars",
        dfcli_version=get_phyloframe_version(),
    )
    add_bool_arg(
        parser,
        "allow-id-reassign",
        default=False,
        help="allow reassignment of ids (default: False)",
    )
    parser.add_argument(
        "--origin-time",
        type=float,
        default=None,
        help="origin time for new root nodes (default: None)",
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
            "phyloframe.legacy._alifestd_prefix_roots_polars",
            logging.info,
        ):
            _run_dataframe_cli(
                base_parser=parser,
                output_dataframe_op=lambda df: alifestd_prefix_roots_polars(
                    df,
                    allow_id_reassign=args.allow_id_reassign,
                    origin_time=args.origin_time,
                    ignore_topological_sensitivity=args.ignore_topological_sensitivity,
                    drop_topological_sensitivity=args.drop_topological_sensitivity,
                ),
            )
    except NotImplementedError as e:
        logging.error(
            "- polars op not yet implemented, use pandas op CLI instead",
        )
        raise e
