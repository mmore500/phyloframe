import argparse
import logging
import os

import joinem
from joinem._dataframe_cli import _add_parser_base, _run_dataframe_cli
import polars as pl

from .._auxlib._add_bool_arg import add_bool_arg
from .._auxlib._begin_prod_logging import begin_prod_logging
from .._auxlib._format_cli_description import format_cli_description
from .._auxlib._get_phyloframe_version import get_phyloframe_version
from .._auxlib._log_context_duration import log_context_duration
from ._alifestd_check_chronological_sensitivity_polars import (
    alifestd_check_chronological_sensitivity_polars,
)


def alifestd_drop_chronological_sensitivity_polars(
    phylogeny_df: pl.DataFrame,
    *,
    shift: bool = True,
    rescale: bool = True,
    reassign: bool = True,
) -> pl.DataFrame:
    """Drop columns from `phylogeny_df` that may be invalidated by
    chronological operations such as coercing chronological consistency.

    Parameters
    ----------
    phylogeny_df : polars.DataFrame
        The phylogeny as a dataframe in alife standard format.
    shift : bool, default True
        Drop columns sensitive to origin time shifts.
    rescale : bool, default True
        Drop columns sensitive to origin time rescaling.
    reassign : bool, default True
        Drop columns sensitive to arbitrary origin time reassignment.

    See Also
    --------
    alifestd_drop_chronological_sensitivity :
        Pandas-based implementation.
    """
    to_drop = alifestd_check_chronological_sensitivity_polars(
        phylogeny_df,
        shift=shift,
        rescale=rescale,
        reassign=reassign,
    )
    return phylogeny_df.drop(to_drop)


_raw_description = f"""\
{os.path.basename(__file__)} | \
(phyloframe v{get_phyloframe_version()}/joinem v{joinem.__version__})

Drop columns that may be invalidated by chronological operations.

Data is assumed to be in alife standard format.

Additional Notes
================
- Use `--eager-read` if modifying data file inplace.

- This CLI entrypoint is experimental and may be subject to change.

See Also
========
phyloframe.legacy._alifestd_drop_chronological_sensitivity :
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
        dfcli_module=(
            "phyloframe.legacy"
            "._alifestd_drop_chronological_sensitivity_polars"
        ),
        dfcli_version=get_phyloframe_version(),
    )
    add_bool_arg(
        parser,
        "shift",
        default=True,
        help="drop columns sensitive to origin time shifts (default: True)",
    )
    add_bool_arg(
        parser,
        "rescale",
        default=True,
        help=(
            "drop columns sensitive to origin time rescaling"
            " (default: True)"
        ),
    )
    add_bool_arg(
        parser,
        "reassign",
        default=True,
        help=(
            "drop columns sensitive to arbitrary origin time reassignment"
            " (default: True)"
        ),
    )
    return parser


if __name__ == "__main__":
    begin_prod_logging()

    parser = _create_parser()
    args, __ = parser.parse_known_args()

    try:
        with log_context_duration(
            "phyloframe.legacy"
            "._alifestd_drop_chronological_sensitivity_polars",
            logging.info,
        ):
            _run_dataframe_cli(
                base_parser=parser,
                output_dataframe_op=(
                    lambda df: alifestd_drop_chronological_sensitivity_polars(
                        df,
                        shift=args.shift,
                        rescale=args.rescale,
                        reassign=args.reassign,
                    )
                ),
            )
    except NotImplementedError as e:
        logging.error(
            "- polars op not yet implemented, use pandas op CLI instead",
        )
        raise e
