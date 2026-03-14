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
from ._alifestd_assign_contiguous_ids_polars import (
    alifestd_assign_contiguous_ids_polars,
)
from ._alifestd_has_contiguous_ids_polars import (
    alifestd_has_contiguous_ids_polars,
)
from ._alifestd_is_topologically_sorted_polars import (
    alifestd_is_topologically_sorted_polars,
)
from ._alifestd_mark_is_right_child_polars import (
    alifestd_mark_is_right_child_polars,
)
from ._alifestd_mark_num_leaves_polars import (
    alifestd_mark_num_leaves_polars,
)
from ._alifestd_mark_num_preceding_leaves_asexual import (
    _alifestd_mark_num_preceding_leaves_asexual_fast_path,
)
from ._alifestd_topological_sort_polars import (
    alifestd_topological_sort_polars,
)
from ._alifestd_try_add_ancestor_id_col_polars import (
    alifestd_try_add_ancestor_id_col_polars,
)


def alifestd_mark_num_preceding_leaves_polars(
    phylogeny_df: pl.DataFrame,
) -> pl.DataFrame:
    """Add column `num_preceding_leaves` with count of all leaves occurring
    before the present node in an inorder traversal.
    """

    logging.info(
        "- alifestd_mark_num_preceding_leaves_polars: adding ancestor_id col...",
    )
    phylogeny_df = alifestd_try_add_ancestor_id_col_polars(phylogeny_df)

    if phylogeny_df.lazy().limit(1).collect().is_empty():
        return phylogeny_df.with_columns(
            num_preceding_leaves=pl.lit(0).cast(pl.Int64),
        )

    logging.info(
        "- alifestd_mark_num_preceding_leaves_polars: checking contiguous ids...",
    )
    if not alifestd_has_contiguous_ids_polars(phylogeny_df):
        phylogeny_df = alifestd_assign_contiguous_ids_polars(phylogeny_df)

    logging.info(
        "- alifestd_mark_num_preceding_leaves_polars: checking topological sort...",
    )
    if not alifestd_is_topologically_sorted_polars(phylogeny_df):
        phylogeny_df = alifestd_topological_sort_polars(phylogeny_df)
        phylogeny_df = alifestd_assign_contiguous_ids_polars(phylogeny_df)

    if "num_leaves" not in phylogeny_df.lazy().collect_schema().names():
        phylogeny_df = alifestd_mark_num_leaves_polars(phylogeny_df)

    if "is_right_child" not in phylogeny_df.lazy().collect_schema().names():
        phylogeny_df = alifestd_mark_is_right_child_polars(phylogeny_df)

    logging.info(
        "- alifestd_mark_num_preceding_leaves_polars: extracting arrays...",
    )
    ancestor_ids = (
        phylogeny_df.lazy()
        .select("ancestor_id")
        .collect()
        .to_series()
        .to_numpy()
    )
    num_leaves = (
        phylogeny_df.lazy()
        .select("num_leaves")
        .collect()
        .to_series()
        .to_numpy()
    )
    is_right_child = (
        phylogeny_df.lazy()
        .select("is_right_child")
        .collect()
        .to_series()
        .to_numpy()
    )

    logging.info(
        "- alifestd_mark_num_preceding_leaves_polars: computing num_preceding_leaves...",
    )
    return phylogeny_df.with_columns(
        num_preceding_leaves=_alifestd_mark_num_preceding_leaves_asexual_fast_path(
            ancestor_ids,
            num_leaves,
            is_right_child,
        ),
    )


_raw_description = f"""{os.path.basename(__file__)} | (phyloframe v{get_phyloframe_version()}/joinem v{joinem.__version__})

Add column `num_preceding_leaves` with count of all leaves occurring before the present node in an inorder traversal.

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
        dfcli_module=(
            "phyloframe.legacy._alifestd_mark_num_preceding_leaves_polars"
        ),
        dfcli_version=get_phyloframe_version(),
    )
    return parser


if __name__ == "__main__":
    begin_prod_logging()

    parser = _create_parser()
    args, __ = parser.parse_known_args()

    try:
        with log_context_duration(
            "phyloframe.legacy._alifestd_mark_num_preceding_leaves_polars",
            logging.info,
        ):
            _run_dataframe_cli(
                base_parser=parser,
                output_dataframe_op=(
                    alifestd_mark_num_preceding_leaves_polars
                ),
            )
    except NotImplementedError as e:
        logging.error(
            "- polars op not yet implemented, use pandas op CLI instead",
        )
        raise e
