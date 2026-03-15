import argparse
import logging
import os

import joinem
from joinem._dataframe_cli import _add_parser_base, _run_dataframe_cli
import polars as pl

from .._auxlib._begin_prod_logging import begin_prod_logging
from .._auxlib._format_cli_description import format_cli_description
from .._auxlib._get_phyloframe_version import get_phyloframe_version
from .._auxlib._log_context_duration import log_context_duration
from ._alifestd_has_contiguous_ids_polars import (
    alifestd_has_contiguous_ids_polars,
)
from ._alifestd_is_topologically_sorted_polars import (
    alifestd_is_topologically_sorted_polars,
)
from ._alifestd_try_add_ancestor_id_col_polars import (
    alifestd_try_add_ancestor_id_col_polars,
)


def alifestd_collapse_trunk_polars(
    phylogeny_df: pl.DataFrame,
) -> pl.DataFrame:
    """Collapse entries masked by `is_trunk` column, keeping only the
    oldest root."""

    if "is_trunk" not in phylogeny_df.lazy().collect_schema().names():
        raise ValueError(
            "`is_trunk` column not provided, trunk is unspecified"
        )

    logging.info(
        "- alifestd_collapse_trunk_polars: adding ancestor_id col...",
    )
    phylogeny_df = alifestd_try_add_ancestor_id_col_polars(phylogeny_df)

    if phylogeny_df.lazy().limit(1).collect().is_empty():
        return phylogeny_df

    if not alifestd_has_contiguous_ids_polars(phylogeny_df):

        raise NotImplementedError(
            "non-contiguous ids not supported",
        )

    if not alifestd_is_topologically_sorted_polars(phylogeny_df):

        raise NotImplementedError(
            "non-topologically-sorted data not supported",
        )

    # Check trunk contiguity: no non-trunk ancestor of trunk entry
    ancestor_is_trunk = pl.col("is_trunk").gather(pl.col("ancestor_id"))
    has_non_contiguous = (
        phylogeny_df.lazy()
        .select(
            (pl.col("is_trunk") & ~ancestor_is_trunk).any(),
        )
        .collect()
        .item()
    )
    if has_non_contiguous:
        raise ValueError("specified trunk is non-contiguous")

    trunk_count = (
        phylogeny_df.lazy().select(pl.col("is_trunk").sum()).collect().item()
    )
    if trunk_count <= 1:
        return phylogeny_df

    logging.info(
        "- alifestd_collapse_trunk_polars: finding oldest trunk root...",
    )
    # Find oldest root among trunk entries (root = ancestor_id == id)
    trunk_roots = phylogeny_df.lazy().filter(
        pl.col("is_trunk") & (pl.col("ancestor_id") == pl.col("id")),
    )
    schema_names = phylogeny_df.lazy().collect_schema().names()
    if "origin_time" in schema_names:
        collapsed_root_id = (
            trunk_roots.sort(["origin_time", "id"])
            .select("id")
            .limit(1)
            .collect()
            .item()
        )
    else:
        collapsed_root_id = (
            trunk_roots.select(pl.col("id").min()).collect().item()
        )

    logging.info(
        "- alifestd_collapse_trunk_polars: collapsing...",
    )
    # Reparent: nodes whose ancestor is trunk -> point to collapsed root
    phylogeny_df = phylogeny_df.with_columns(
        ancestor_id=pl.when(ancestor_is_trunk)
        .then(pl.lit(collapsed_root_id))
        .otherwise(pl.col("ancestor_id")),
    )

    # Keep non-trunk entries plus the collapsed root
    return phylogeny_df.filter(
        ~pl.col("is_trunk") | (pl.col("id") == collapsed_root_id),
    )


_raw_description = f"""{os.path.basename(__file__)} | (phyloframe v{get_phyloframe_version()}/joinem v{joinem.__version__})

Collapse entries masked by `is_trunk` column, keeping only the oldest root.

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
        dfcli_module=("phyloframe.legacy._alifestd_collapse_trunk_polars"),
        dfcli_version=get_phyloframe_version(),
    )
    return parser


if __name__ == "__main__":
    begin_prod_logging()

    parser = _create_parser()
    args, __ = parser.parse_known_args()

    try:
        with log_context_duration(
            "phyloframe.legacy._alifestd_collapse_trunk_polars",
            logging.info,
        ):
            _run_dataframe_cli(
                base_parser=parser,
                output_dataframe_op=alifestd_collapse_trunk_polars,
            )
    except NotImplementedError as e:
        logging.error(
            "- polars op not yet implemented, use pandas op CLI instead",
        )
        raise e
