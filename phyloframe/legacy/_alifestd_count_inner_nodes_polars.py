import argparse
import logging
import os

import polars as pl

from .._auxlib._begin_prod_logging import begin_prod_logging
from .._auxlib._format_cli_description import format_cli_description
from .._auxlib._get_phyloframe_version import get_phyloframe_version
from .._auxlib._log_context_duration import log_context_duration
from ._alifestd_count_leaf_nodes_polars import alifestd_count_leaf_nodes_polars


def alifestd_count_inner_nodes_polars(phylogeny_df: pl.DataFrame) -> int:
    """Count how many non-leaf nodes are contained in phylogeny."""
    num_leaves = alifestd_count_leaf_nodes_polars(phylogeny_df)
    num_total = phylogeny_df.lazy().select(pl.len()).collect().item()
    res = num_total - num_leaves
    assert res >= 0
    return res


_raw_description = f"""{os.path.basename(__file__)} | (phyloframe v{get_phyloframe_version()})

Print number of inner (non-leaf) nodes in alife-standard phylogeny.

Note that this CLI entrypoint is experimental and may be subject to change.
"""


def _create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=format_cli_description(_raw_description),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "phylogeny_file",
        type=str,
        help="Alife standard dataframe file to process.",
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=get_phyloframe_version(),
    )
    return parser


if __name__ == "__main__":
    begin_prod_logging()

    parser = _create_parser()
    args = parser.parse_args()
    input_ext = os.path.splitext(args.phylogeny_file)[1]

    logging.info(
        f"reading alife-standard {input_ext} phylogeny data from "
        f"{args.phylogeny_file}...",
    )
    phylogeny_df = {
        ".csv": pl.read_csv,
        ".fea": pl.read_ipc,
        ".feather": pl.read_ipc,
        ".pqt": pl.read_parquet,
        ".parquet": pl.read_parquet,
    }[input_ext](args.phylogeny_file)

    with log_context_duration(
        "phyloframe.legacy.alifestd_count_inner_nodes_polars", logging.info
    ):
        count = alifestd_count_inner_nodes_polars(phylogeny_df)

    print(count)

    logging.info("done!")
