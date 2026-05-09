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
from ._alifestd_mark_leaves import alifestd_mark_leaves


def alifestd_ultrametricize(
    phylogeny_df: pd.DataFrame,
    mutate: bool = False,
    *,
    method: str = "extend",
) -> pd.DataFrame:
    """Adjust tip `origin_time` values so all tips share the same time.

    With ``method="extend"``, each tip's ``origin_time`` is set to the
    maximum ``origin_time`` among tips. Internal node times are not
    modified.

    Empty phylogenies are returned unchanged.

    Input dataframe is not mutated by this operation unless `mutate` set True.
    If mutate set True, operation does not occur in place; still use return
    value to get transformed phylogeny dataframe.
    """
    if "origin_time" not in phylogeny_df.columns:
        raise ValueError(
            "alifestd_ultrametricize requires 'origin_time' column",
        )

    if method != "extend":
        raise ValueError(
            f"alifestd_ultrametricize: unknown method {method!r}",
        )

    if not mutate:
        phylogeny_df = phylogeny_df.copy()

    if phylogeny_df.empty:
        return phylogeny_df

    if "is_leaf" not in phylogeny_df.columns:
        phylogeny_df = alifestd_mark_leaves(phylogeny_df, mutate=True)

    leaf_mask = phylogeny_df["is_leaf"].to_numpy()
    target_origin_time = phylogeny_df.loc[leaf_mask, "origin_time"].max()
    phylogeny_df.loc[leaf_mask, "origin_time"] = target_origin_time

    return phylogeny_df


_raw_description = f"""{os.path.basename(__file__)} | (phyloframe v{get_phyloframe_version()}/joinem v{joinem.__version__})

Adjust tip `origin_time` values so all tips share the same time.

With method "extend", each tip's `origin_time` is set to the maximum
`origin_time` among tips. Internal node times are not modified.

Data is assumed to be in alife standard format.

Additional Notes
================
- Use `--eager-read` if modifying data file inplace.

- This CLI entrypoint is experimental and may be subject to change.

See Also
========
phyloframe.legacy._alifestd_ultrametricize_polars :
    Entrypoint for high-performance Polars-based implementation.
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
        dfcli_module="phyloframe.legacy._alifestd_ultrametricize",
        dfcli_version=get_phyloframe_version(),
    )
    parser.add_argument(
        "--method",
        default="extend",
        type=str,
        help="ultrametricization method (default: extend)",
    )
    return parser


if __name__ == "__main__":
    begin_prod_logging()

    parser = _create_parser()
    args, __ = parser.parse_known_args()
    with log_context_duration(
        "phyloframe.legacy._alifestd_ultrametricize", logging.info
    ):
        _run_dataframe_cli(
            base_parser=parser,
            output_dataframe_op=delegate_polars_implementation()(
                functools.partial(alifestd_ultrametricize, method=args.method),
            ),
        )
