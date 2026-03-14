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
from ._alifestd_add_inner_knuckles_polars import (
    alifestd_add_inner_knuckles_polars,
)
from ._alifestd_mark_leaves_polars import alifestd_mark_leaves_polars
from ._alifestd_mark_node_depth_polars import alifestd_mark_node_depth_polars


def alifestd_add_inner_niblings_polars(
    phylogeny_df: pl.DataFrame,
) -> pl.DataFrame:
    """For all inner nodes, add a subtending unifurcation, adding a "nibling"
    leaf as the child of the knuckle.

    Here, "nibling" refers to a leaf that is a niece/nephew of the inner node.

    Parameters
    ----------
    phylogeny_df : polars.DataFrame or polars.LazyFrame
        The phylogeny as a dataframe in alife standard format.

        Must represent an asexual phylogeny with contiguous ids and
        topologically sorted rows.

    Returns
    -------
    polars.DataFrame
        The phylogeny with inner niblings added.

    See Also
    --------
    alifestd_add_inner_niblings_asexual :
        Pandas-based implementation.
    """
    if isinstance(phylogeny_df, pl.LazyFrame):
        phylogeny_df = phylogeny_df.collect()

    if phylogeny_df.lazy().limit(1).collect().is_empty():
        return phylogeny_df

    logging.info(
        "- alifestd_add_inner_niblings_polars: adding inner knuckles...",
    )
    phylogeny_df = alifestd_add_inner_knuckles_polars(phylogeny_df)

    logging.info(
        "- alifestd_add_inner_niblings_polars: marking node depth...",
    )
    phylogeny_df = alifestd_mark_node_depth_polars(phylogeny_df)

    logging.info(
        "- alifestd_add_inner_niblings_polars: marking leaves...",
    )
    phylogeny_df = alifestd_mark_leaves_polars(phylogeny_df)

    logging.info(
        "- alifestd_add_inner_niblings_polars: creating niblings...",
    )
    nibling_mask = (pl.col("node_depth") % 2 == 1) & ~pl.col("is_leaf")
    nibling_df = phylogeny_df.filter(nibling_mask)

    if nibling_df.is_empty():
        return phylogeny_df

    id_delta = phylogeny_df.select(pl.col("id").max()).item() + 1
    nibling_df = nibling_df.with_columns(
        (pl.col("id") + id_delta).alias("id"),
        pl.lit(True).alias("is_leaf"),
    )

    if (
        nibling_df.select(pl.col("id").min()).item()
        <= phylogeny_df.select(pl.col("id").max()).item()
    ):
        raise ValueError("overflow in new id assigment")

    return pl.concat([phylogeny_df, nibling_df], how="diagonal")


_raw_description = f"""\
{os.path.basename(__file__)} | \
(phyloframe v{get_phyloframe_version()}/joinem v{joinem.__version__})

For all inner nodes, add a subtending unifurcation, adding a nibling leaf \
as the child of the knuckle.

Data is assumed to be in alife standard format.

Additional Notes
================
- Requires 'ancestor_id' column (or 'ancestor_list' column from which \
ancestor_id can be derived).

- Use `--eager-read` if modifying data file inplace.

- This CLI entrypoint is experimental and may be subject to change.

See Also
========
phyloframe.legacy._alifestd_add_inner_niblings_asexual :
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
        dfcli_module=("phyloframe.legacy._alifestd_add_inner_niblings_polars"),
        dfcli_version=get_phyloframe_version(),
    )
    return parser


if __name__ == "__main__":
    begin_prod_logging()

    parser = _create_parser()
    args, __ = parser.parse_known_args()

    try:
        with log_context_duration(
            "phyloframe.legacy._alifestd_add_inner_niblings_polars",
            logging.info,
        ):
            _run_dataframe_cli(
                base_parser=parser,
                output_dataframe_op=(alifestd_add_inner_niblings_polars),
            )
    except NotImplementedError as e:
        logging.error(
            "- polars op not yet implemented, use pandas op CLI instead",
        )
        raise e
