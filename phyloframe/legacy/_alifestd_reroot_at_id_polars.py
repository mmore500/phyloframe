import argparse
import functools
import logging
import os

import joinem
from joinem._dataframe_cli import _add_parser_base, _run_dataframe_cli
import polars as pl

from .._auxlib._begin_prod_logging import begin_prod_logging
from .._auxlib._format_cli_description import format_cli_description
from .._auxlib._get_phyloframe_version import get_phyloframe_version
from .._auxlib._log_context_duration import log_context_duration
from .._auxlib._unfurl_lineage_with_contiguous_ids import (
    unfurl_lineage_with_contiguous_ids,
)
from ._alifestd_has_contiguous_ids_polars import (
    alifestd_has_contiguous_ids_polars,
)
from ._alifestd_is_topologically_sorted_polars import (
    alifestd_is_topologically_sorted_polars,
)
from ._alifestd_make_ancestor_list_col_polars import (
    alifestd_make_ancestor_list_col_polars,
)
from ._alifestd_try_add_ancestor_id_col_polars import (
    alifestd_try_add_ancestor_id_col_polars,
)


def alifestd_reroot_at_id_polars(
    phylogeny_df: pl.DataFrame,
    new_root_id: int,
) -> pl.DataFrame:
    """Reroot phylogeny at specified node id, preserving topology.

    Reverses the descendant-to-ancestor relationships of all ancestors of the
    new root. Does not update branch_lengths or edge_lengths columns if
    present.

    Parameters
    ----------
    phylogeny_df : polars.DataFrame
        The phylogeny as a dataframe in alife standard format.

        Must represent an asexual phylogeny.
    new_root_id : int
        The ID of the node to use as the new root of the phylogeny.

    Returns
    -------
    polars.DataFrame
        The rerooted phylogeny in alife standard format.
    """
    phylogeny_df = alifestd_try_add_ancestor_id_col_polars(phylogeny_df)

    # Mark the target node before any id reassignment
    phylogeny_df = phylogeny_df.with_columns(
        __is_new_root__=(pl.col("id") == new_root_id),
    )

    if not alifestd_has_contiguous_ids_polars(phylogeny_df):

        raise NotImplementedError(
            "non-contiguous ids not supported",
        )

    if not alifestd_is_topologically_sorted_polars(phylogeny_df):

        raise NotImplementedError(
            "non-topologically-sorted data not supported",
        )

    # Look up new_root_id after any id reassignment
    new_root_id = int(
        phylogeny_df.lazy()
        .filter(pl.col("__is_new_root__"))
        .select("id")
        .collect()
        .item()
    )
    phylogeny_df = phylogeny_df.drop("__is_new_root__")

    logging.info(
        "- alifestd_reroot_at_id_polars: unfurling lineage...",
    )
    ancestor_ids = (
        phylogeny_df.lazy()
        .select("ancestor_id")
        .collect()
        .to_series()
        .to_numpy()
        .copy()
    )
    lineage = unfurl_lineage_with_contiguous_ids(ancestor_ids, new_root_id)

    logging.info(
        "- alifestd_reroot_at_id_polars: reversing ancestor relationships...",
    )
    # lineage = [new_root, parent_of_new_root, grandparent, ..., old_root]
    # For each parent in the lineage, set its ancestor_id to its child
    copy_to_slice = lineage[1:]  # parents, grandparents, ..., old_root
    copy_from_slice = lineage[:-1]  # children
    ids = phylogeny_df.lazy().select("id").collect().to_series().to_numpy()
    ancestor_ids[copy_to_slice] = ids[copy_from_slice]
    ancestor_ids[new_root_id] = ids[new_root_id]

    logging.info(
        "- alifestd_reroot_at_id_polars: updating ancestor_id column...",
    )
    phylogeny_df = phylogeny_df.with_columns(
        ancestor_id=pl.Series("ancestor_id", ancestor_ids),
    )

    if "ancestor_list" in phylogeny_df.lazy().collect_schema().names():
        logging.info(
            "- alifestd_reroot_at_id_polars: recomputing ancestor_list...",
        )
        phylogeny_df = phylogeny_df.with_columns(
            ancestor_list=alifestd_make_ancestor_list_col_polars(
                phylogeny_df["id"],
                phylogeny_df["ancestor_id"],
            ),
        )

    return phylogeny_df


_raw_description = f"""{os.path.basename(__file__)} | (phyloframe v{get_phyloframe_version()}/joinem v{joinem.__version__})

Reroot phylogeny at specified node id, preserving topology.

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
        dfcli_module="phyloframe.legacy._alifestd_reroot_at_id_polars",
        dfcli_version=get_phyloframe_version(),
    )
    parser.add_argument(
        "--new-root-id",
        type=int,
        required=True,
        help="id of the node to reroot at",
    )
    return parser


if __name__ == "__main__":
    begin_prod_logging()

    parser = _create_parser()
    args, __ = parser.parse_known_args()

    try:
        with log_context_duration(
            "phyloframe.legacy._alifestd_reroot_at_id_polars",
            logging.info,
        ):
            _run_dataframe_cli(
                base_parser=parser,
                output_dataframe_op=functools.partial(
                    alifestd_reroot_at_id_polars,
                    new_root_id=args.new_root_id,
                ),
            )
    except NotImplementedError as e:
        logging.error(
            "- polars op not yet implemented, use pandas op CLI instead",
        )
        raise e
