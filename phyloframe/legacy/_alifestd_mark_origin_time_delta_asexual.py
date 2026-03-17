import argparse
import functools
import logging
import os

import joinem
from joinem._dataframe_cli import _add_parser_base, _run_dataframe_cli
import pandas as pd

from .._auxlib._begin_prod_logging import begin_prod_logging
from .._auxlib._delegate_polars_implementation import (
    delegate_polars_implementation,
)
from .._auxlib._format_cli_description import format_cli_description
from .._auxlib._get_phyloframe_version import get_phyloframe_version
from .._auxlib._log_context_duration import log_context_duration
from ._alifestd_mark_ancestor_origin_time_asexual import (
    alifestd_mark_ancestor_origin_time_asexual,
)


def alifestd_mark_origin_time_delta_asexual(
    phylogeny_df: pd.DataFrame,
    mutate: bool = False,
    mark_as: str = "origin_time_delta",
) -> pd.DataFrame:
    """Add columns `origin_time_delta` and `ancestor_origin_time`.

    The output column name can be changed via the ``mark_as`` parameter.

    Dataframe must provide column `origin_time`.

    A topological sort will be applied if `phylogeny_df` is not topologically
    sorted. Dataframe reindexing (e.g., df.index) may be applied.

    Input dataframe is not mutated by this operation unless `mutate` set True.
    If mutate set True, operation does not occur in place; still use return
    value to get transformed phylogeny dataframe.
    """
    if not mutate:
        phylogeny_df = phylogeny_df.copy()

    phylogeny_df = alifestd_mark_ancestor_origin_time_asexual(
        phylogeny_df, mutate=True
    )

    phylogeny_df[mark_as] = (
        phylogeny_df["origin_time"] - phylogeny_df["ancestor_origin_time"]
    )

    return phylogeny_df


_raw_description = f"""{os.path.basename(__file__)} | (phyloframe v{get_phyloframe_version()}/joinem v{joinem.__version__})

Add columns `origin_time_delta` and `ancestor_origin_time`.

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
        dfcli_module="phyloframe.legacy._alifestd_mark_origin_time_delta_asexual",
        dfcli_version=get_phyloframe_version(),
    )
    parser.add_argument(
        "--mark-as",
        default="origin_time_delta",
        type=str,
        help="output column name (default: origin_time_delta)",
    )
    return parser


if __name__ == "__main__":
    begin_prod_logging()

    parser = _create_parser()
    args, __ = parser.parse_known_args()
    with log_context_duration(
        "phyloframe.legacy._alifestd_mark_origin_time_delta_asexual",
        logging.info,
    ):
        _run_dataframe_cli(
            base_parser=parser,
            output_dataframe_op=delegate_polars_implementation()(
                functools.partial(
                    alifestd_mark_origin_time_delta_asexual,
                    mark_as=args.mark_as,
                ),
            ),
        )
