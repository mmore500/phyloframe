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
from ._alifestd_make_ancestor_list_col_polars import (
    alifestd_make_ancestor_list_col_polars,
)
from ._alifestd_mark_leaves_polars import alifestd_mark_leaves_polars
from ._alifestd_try_add_ancestor_id_col_polars import (
    alifestd_try_add_ancestor_id_col_polars,
)


def alifestd_add_inner_knuckles_polars(
    phylogeny_df: pl.DataFrame,
) -> pl.DataFrame:
    """For all inner nodes, add a subtending unifurcation ("knuckle").

    Parameters
    ----------
    phylogeny_df : polars.DataFrame
        The phylogeny as a dataframe in alife standard format.

        Must represent an asexual phylogeny with contiguous ids and
        topological sort order.

    Returns
    -------
    polars.DataFrame
        The phylogeny with knuckle nodes added for each inner node.

    See Also
    --------
    alifestd_add_inner_knuckles_asexual :
        Pandas-based implementation.
    """
    if isinstance(phylogeny_df, pl.LazyFrame):
        phylogeny_df = phylogeny_df.collect()

    if phylogeny_df.lazy().limit(1).collect().is_empty():
        return phylogeny_df

    phylogeny_df = alifestd_try_add_ancestor_id_col_polars(phylogeny_df)
    if "ancestor_id" not in phylogeny_df.lazy().collect_schema().names():
        raise ValueError("asexual phylogeny required")

    if not alifestd_has_contiguous_ids_polars(phylogeny_df):
        phylogeny_df = alifestd_assign_contiguous_ids_polars(phylogeny_df)

    if not alifestd_is_topologically_sorted_polars(phylogeny_df):
        phylogeny_df = alifestd_topological_sort_polars(phylogeny_df)
        phylogeny_df = alifestd_assign_contiguous_ids_polars(phylogeny_df)

    if "is_leaf" not in phylogeny_df.lazy().collect_schema().names():
        phylogeny_df = alifestd_mark_leaves_polars(phylogeny_df)

    inner_df = phylogeny_df.filter(~pl.col("is_leaf"))

    if inner_df.lazy().limit(1).collect().is_empty():
        return phylogeny_df

    id_delta = phylogeny_df.select(pl.col("id").max()).item() + 1

    # Build knuckle copies of inner nodes:
    # - New id = old id + id_delta
    # - For root nodes (id == ancestor_id), ancestor_id also shifts by id_delta
    # - For non-root inner nodes, ancestor_id stays as-is
    # - is_root (if present) set to False
    # - origin_time_delta (if present) set to 0
    knuckle_exprs = [
        (pl.col("id") + id_delta).alias("id"),
        pl.when(pl.col("id") == pl.col("ancestor_id"))
        .then(pl.col("ancestor_id") + id_delta)
        .otherwise(pl.col("ancestor_id"))
        .alias("ancestor_id"),
    ]

    knuckle_df = inner_df.with_columns(knuckle_exprs)

    if "is_root" in knuckle_df.lazy().collect_schema().names():
        knuckle_df = knuckle_df.with_columns(
            is_root=pl.lit(False),
        )

    if "origin_time_delta" in knuckle_df.lazy().collect_schema().names():
        knuckle_df = knuckle_df.with_columns(
            origin_time_delta=pl.lit(0),
        )

    # Verify no overflow / id collision
    knuckle_id_min = knuckle_df.select(pl.col("id").min()).item()
    orig_id_max = phylogeny_df.select(pl.col("id").max()).item()
    if not (knuckle_id_min > orig_id_max):
        raise ValueError("overflow in new id assignment")

    # Update original inner nodes' ancestor_id to point to their knuckle
    phylogeny_df = phylogeny_df.with_columns(
        pl.when(~pl.col("is_leaf"))
        .then(pl.col("id") + id_delta)
        .otherwise(pl.col("ancestor_id"))
        .alias("ancestor_id"),
    )

    res = pl.concat([phylogeny_df, knuckle_df], how="diagonal")

    if "ancestor_list" in res.lazy().collect_schema().names():
        res = res.with_columns(
            ancestor_list=alifestd_make_ancestor_list_col_polars(
                res["id"],
                res["ancestor_id"],
            ),
        )

    return res


_raw_description = f"""{os.path.basename(__file__)} | (phyloframe v{get_phyloframe_version()}/joinem v{joinem.__version__})

For all inner nodes, add a subtending unifurcation (knuckle).

Data is assumed to be in alife standard format.

Additional Notes
================
- Requires contiguous ids and topological sort order.

- Use `--eager-read` if modifying data file inplace.

- This CLI entrypoint is experimental and may be subject to change.

See Also
========
phyloframe.legacy._alifestd_add_inner_knuckles_asexual :
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
        dfcli_module="phyloframe.legacy._alifestd_add_inner_knuckles_polars",
        dfcli_version=get_phyloframe_version(),
    )
    return parser


if __name__ == "__main__":
    begin_prod_logging()

    parser = _create_parser()
    args, __ = parser.parse_known_args()

    try:
        with log_context_duration(
            "phyloframe.legacy._alifestd_add_inner_knuckles_polars",
            logging.info,
        ):
            _run_dataframe_cli(
                base_parser=parser,
                output_dataframe_op=alifestd_add_inner_knuckles_polars,
            )
    except NotImplementedError as e:
        logging.error(
            "- polars op not yet implemented, use pandas op CLI instead",
        )
        raise e
