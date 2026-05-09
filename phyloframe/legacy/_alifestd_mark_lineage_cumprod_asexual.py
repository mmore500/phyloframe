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
from ._alifestd_mark_lineage_cumsum_asexual import (
    _OP_PROD,
    _alifestd_mark_lineage_op_asexual,
)


def alifestd_mark_lineage_cumprod_asexual(
    phylogeny_df: pd.DataFrame,
    values: str,
    mutate: bool = False,
    *,
    mark_as: str = "lineage_cumprod",
    reverse: bool = False,
    skipna: bool = True,
) -> pd.DataFrame:
    """Add column with cumulative product of ``values`` along each lineage.

    With ``reverse=False`` (default), the result at each node is the
    product of ``values`` along the path from the root to that node,
    inclusive. With ``reverse=True``, the result at each node is the
    product of ``values`` over the entire clade rooted at that node,
    inclusive.

    The output column name can be changed via the ``mark_as`` parameter.
    NaN values are treated as 1 if ``skipna`` (default), else propagate.

    Phylogeny must be asexual, topologically sorted, and have
    contiguous ids; otherwise ``NotImplementedError`` is raised.

    Input dataframe is not mutated by this operation unless ``mutate``
    is set True. If mutate set True, operation does not occur in place;
    still use return value to get transformed phylogeny dataframe.
    """
    return _alifestd_mark_lineage_op_asexual(
        phylogeny_df,
        values,
        _OP_PROD,
        mutate=mutate,
        mark_as=mark_as,
        reverse=reverse,
        skipna=skipna,
    )


_raw_description = f"""{os.path.basename(__file__)} | (phyloframe v{get_phyloframe_version()}/joinem v{joinem.__version__})

Add column with cumulative product of `values` along each lineage.

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
        dfcli_module="phyloframe.legacy._alifestd_mark_lineage_cumprod_asexual",
        dfcli_version=get_phyloframe_version(),
    )
    parser.add_argument(
        "--values",
        default=None,
        type=str,
        help="name of column to aggregate along lineage",
    )
    parser.add_argument(
        "--mark-as",
        default="lineage_cumprod",
        type=str,
        help="output column name (default: lineage_cumprod)",
    )
    parser.add_argument(
        "--reverse",
        action="store_true",
        help="aggregate over clade rooted at each node "
        "instead of along lineage from root",
    )
    parser.add_argument(
        "--no-skipna",
        action="store_true",
        help="propagate NaN values rather than treating as identity",
    )
    return parser


if __name__ == "__main__":
    begin_prod_logging()

    parser = _create_parser()
    args, __ = parser.parse_known_args()
    with log_context_duration(
        "phyloframe.legacy._alifestd_mark_lineage_cumprod_asexual",
        logging.info,
    ):
        _run_dataframe_cli(
            base_parser=parser,
            output_dataframe_op=delegate_polars_implementation()(
                functools.partial(
                    alifestd_mark_lineage_cumprod_asexual,
                    values=args.values,
                    mark_as=args.mark_as,
                    reverse=args.reverse,
                    skipna=not args.no_skipna,
                ),
            ),
        )
