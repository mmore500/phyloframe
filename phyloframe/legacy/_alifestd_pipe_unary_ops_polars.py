import argparse
import importlib
import logging
import os
import typing

import joinem
from joinem._dataframe_cli import _add_parser_base, _run_dataframe_cli
import numpy as np
import opytional as opyt
import pandas as pd
import polars as pl

from .._auxlib._begin_prod_logging import begin_prod_logging
from .._auxlib._format_cli_description import format_cli_description
from .._auxlib._get_phyloframe_version import get_phyloframe_version
from .._auxlib._log_context_duration import log_context_duration


def alifestd_pipe_unary_ops_polars(
    phylogeny_df: pl.DataFrame,
    *unary_ops: typing.Callable[[pl.DataFrame], pl.DataFrame],
    progress_wrap: typing.Callable = lambda x: x,
) -> pl.DataFrame:
    """Pipe a phylogeny DataFrame through a sequence of unary operations.

    Each operation in `unary_ops` is applied in order to the DataFrame.

    Parameters
    ----------
    phylogeny_df : polars.DataFrame
        The phylogeny as a dataframe in alife standard format.
    *unary_ops : callable
        Zero or more callables, each accepting and returning a DataFrame.
    progress_wrap : callable, optional
        Optional wrapper for unary_ops to provide progress feedback (e.g. tqdm).

    Returns
    -------
    polars.DataFrame
        The result of piping `phylogeny_df` through each operation in order.

    See Also
    --------
    alifestd_pipe_unary_ops :
        Pandas-based implementation.
    """
    for i, op in enumerate(progress_wrap(unary_ops)):
        logging.info(
            "- alifestd_pipe_unary_ops_polars: "
            f"applying op {i + 1} of {len(unary_ops)} {op!r}...",
        )
        phylogeny_df = op(phylogeny_df)
    return phylogeny_df


_raw_description = f"""{os.path.basename(__file__)} | (phyloframe v{get_phyloframe_version()}/joinem v{joinem.__version__})

Pipe a phylogeny DataFrame through a sequence of unary operations.

Each --op flag is a Python expression that evaluates to a callable
accepting and returning a DataFrame. Operations are applied in the order
they are provided.

The following names are available in the eval context:
  pf        : phyloframe
  pfl       : phyloframe.legacy
  pd        : pandas
  pl        : polars
  np        : numpy
  opyt      : opytional
  importlib : importlib

Data is assumed to be in alife standard format.

Additional Notes
================
- Use `--eager-read` if modifying data file inplace.

- This CLI entrypoint is experimental and may be subject to change.

See Also
========
phyloframe.legacy._alifestd_pipe_unary_ops :
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
        dfcli_module="phyloframe.legacy._alifestd_pipe_unary_ops_polars",
        dfcli_version=get_phyloframe_version(),
    )
    parser.add_argument(
        "--op",
        action="append",
        default=[],
        dest="ops",
        help=(
            "Python expression evaluating to a unary DataFrame op. "
            "Specify multiple ops by using this flag multiple times. "
            "Example: 'lambda df: df.with_columns(pl.col(\"my_col\") * 2)'"
        ),
        metavar="EXPR",
    )
    return parser


def _build_eval_context() -> dict:
    return {
        "importlib": importlib,
        "np": np,
        "opyt": opyt,
        "pd": pd,
        "pf": importlib.import_module("phyloframe"),
        "pfl": importlib.import_module("phyloframe.legacy"),
        "pl": pl,
    }


if __name__ == "__main__":
    begin_prod_logging()

    parser = _create_parser()
    args, __ = parser.parse_known_args()

    eval_context = _build_eval_context()
    unary_ops = [eval(op, eval_context) for op in args.ops]  # nosec B307

    try:
        with log_context_duration(
            "phyloframe.legacy._alifestd_pipe_unary_ops_polars",
            logging.info,
        ):
            _run_dataframe_cli(
                base_parser=parser,
                output_dataframe_op=lambda df: alifestd_pipe_unary_ops_polars(
                    df, *unary_ops
                ),
            )
    except NotImplementedError as e:
        logging.error(
            "- polars op not yet implemented, use pandas op CLI instead",
        )
        raise e
