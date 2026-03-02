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
from ._alifestd_convert_root_ancestor_token import (
    alifestd_convert_root_ancestor_token,
)


def alifestd_assign_root_ancestor_token(
    phylogeny_df: pd.DataFrame,
    root_ancestor_token: str,
    mutate: bool = False,
) -> pd.DataFrame:
    """Set `root_ancestor_token` for "ancestor_list" column.

    The option `root_ancestor_token` will be sandwiched in brackets to create
    the ancestor list entry for genesis organisms. For example, the token
    "None" will yield the entry "[None]" and the token "" will yield the entry
    """

    if not mutate:
        phylogeny_df = phylogeny_df.copy()

    phylogeny_df["ancestor_list"] = alifestd_convert_root_ancestor_token(
        phylogeny_df["ancestor_list"],
        root_ancestor_token,
        mutate=False,  # prevent assign to slice warning
    )

    return phylogeny_df


_raw_description = f"""{os.path.basename(__file__)} | (phyloframe v{get_phyloframe_version()}/joinem v{joinem.__version__})

Set `root_ancestor_token` for ancestor_list column.

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
        dfcli_module="phyloframe.legacy._alifestd_assign_root_ancestor_token",
        dfcli_version=get_phyloframe_version(),
    )
    parser.add_argument(
        "--root-ancestor-token",
        type=str,
        default="none",
        help='token for root ancestor entries (default: "none")',
    )
    return parser


if __name__ == "__main__":
    begin_prod_logging()

    parser = _create_parser()
    args, __ = parser.parse_known_args()
    with log_context_duration(
        "phyloframe.legacy._alifestd_assign_root_ancestor_token",
        logging.info,
    ):
        _run_dataframe_cli(
            base_parser=parser,
            output_dataframe_op=delegate_polars_implementation()(
                functools.partial(
                    alifestd_assign_root_ancestor_token,
                    root_ancestor_token=args.root_ancestor_token,
                ),
            ),
        )
