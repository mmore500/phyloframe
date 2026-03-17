import argparse
import contextlib
import functools
import gc
import logging
import os
import sys
import typing

import joinem
from joinem._dataframe_cli import _add_parser_base, _run_dataframe_cli
import numpy as np
import opytional as opyt
import polars as pl

from .._auxlib._RngStateContext import RngStateContext
from .._auxlib._add_bool_arg import add_bool_arg
from .._auxlib._begin_prod_logging import begin_prod_logging
from .._auxlib._format_cli_description import format_cli_description
from .._auxlib._get_phyloframe_version import get_phyloframe_version
from .._auxlib._log_context_duration import log_context_duration
from .._auxlib._log_memory_usage import log_memory_usage
from ._alifestd_find_leaf_ids_polars import alifestd_find_leaf_ids_polars
from ._alifestd_try_add_ancestor_id_col_polars import (
    alifestd_try_add_ancestor_id_col_polars,
)


def _alifestd_mark_sample_tips_polars_impl(
    phylogeny_df: pl.DataFrame,
    n_sample: int,
    mark_as: str,
) -> pl.DataFrame:
    """Implementation detail for alifestd_mark_sample_tips_polars."""

    logging.info(
        "- alifestd_mark_sample_tips_polars: collecting leaf ids...",
    )
    leaf_ids = alifestd_find_leaf_ids_polars(phylogeny_df)
    gc.collect()
    log_memory_usage(logging.info)

    logging.info(
        "- alifestd_mark_sample_tips_polars: sampling leaf_ids...",
    )
    leaf_ids = np.random.choice(
        leaf_ids, size=min(n_sample, len(leaf_ids)), replace=False
    )
    gc.collect()
    log_memory_usage(logging.info)

    logging.info("- alifestd_mark_sample_tips_polars: collecting len(df)...")
    len_df = phylogeny_df.lazy().select(pl.len()).collect().item()

    logging.info("- alifestd_mark_sample_tips_polars: finding marked...")
    marked_mask = np.bincount(leaf_ids, minlength=len_df).astype(bool)
    del leaf_ids
    gc.collect()
    log_memory_usage(logging.info)

    logging.info(
        "- alifestd_mark_sample_tips_polars: setting mark column...",
    )
    phylogeny_df = phylogeny_df.with_columns(
        pl.Series(name=mark_as, values=marked_mask),
    )
    gc.collect()
    log_memory_usage(logging.info)

    return phylogeny_df


def alifestd_mark_sample_tips_polars(
    phylogeny_df: pl.DataFrame,
    n_sample: int,
    seed: typing.Optional[int] = None,
    mark_as: str = "alifestd_mark_sample_tips_polars",
) -> pl.DataFrame:
    """Mark a random subsample of `n_sample` tips.

    Adds a boolean column ``mark_as`` indicating retained tips.

    If `n_sample` is greater than the number of tips in the phylogeny,
    all tips are marked.

    Only supports asexual phylogenies.

    Parameters
    ----------
    phylogeny_df : polars.DataFrame
        The phylogeny as a dataframe in alife standard format.

        Must represent an asexual phylogeny.
    n_sample : int
        Number of tips to mark.
    seed : int, optional
        Integer seed for deterministic behavior.
    mark_as : str, default "alifestd_mark_sample_tips_polars"
        Column name for the boolean mark.

    Raises
    ------
    NotImplementedError
        If `phylogeny_df` has no "ancestor_id" column.

    Returns
    -------
    polars.DataFrame
        The phylogeny with an added boolean mark column.

    See Also
    --------
    alifestd_mark_sample_tips_asexual :
        Pandas-based implementation.
    """
    phylogeny_df = alifestd_try_add_ancestor_id_col_polars(phylogeny_df)

    if phylogeny_df.lazy().limit(1).collect().is_empty():
        return phylogeny_df.with_columns(
            pl.lit(False).alias(mark_as),
        )

    with opyt.apply_if_or_else(seed, RngStateContext, contextlib.nullcontext):
        return _alifestd_mark_sample_tips_polars_impl(
            phylogeny_df,
            n_sample,
            mark_as,
        )


_raw_description = f"""{os.path.basename(__file__)} | (phyloframe v{get_phyloframe_version()}/joinem v{joinem.__version__})

Mark a random subsample of `-n` tips with a boolean column.

If `-n` is greater than the number of tips in the phylogeny, all tips are marked.

Data is assumed to be in alife standard format.
Only supports asexual phylogenies.

Additional Notes
================
- Requires 'ancestor_id' column to be present in input DataFrame.
Otherwise, no action is taken.

- Use `--eager-read` if modifying data file inplace.

- This CLI entrypoint is experimental and may be subject to change.

See Also
========
phyloframe.legacy._alifestd_mark_sample_tips_asexual :
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
        dfcli_module="phyloframe.legacy._alifestd_mark_sample_tips_polars",
        dfcli_version=get_phyloframe_version(),
    )
    parser.add_argument(
        "-n",
        default=sys.maxsize,
        type=int,
        help="Number of tips to mark.",
    )
    parser.add_argument(
        "--seed",
        default=None,
        dest="seed",
        help="Integer seed for deterministic behavior.",
        type=int,
    )
    parser.add_argument(
        "--mark-as",
        default="alifestd_mark_sample_tips_polars",
        type=str,
        help="Column name for the boolean mark (default: alifestd_mark_sample_tips_polars).",
    )
    add_bool_arg(
        parser,
        "ignore-topological-sensitivity",
        default=False,
        help="suppress topological sensitivity warning (default: False)",
    )
    add_bool_arg(
        parser,
        "drop-topological-sensitivity",
        default=False,
        help="drop topology-sensitive columns from output (default: False)",
    )
    return parser


if __name__ == "__main__":
    begin_prod_logging()

    parser = _create_parser()
    args, __ = parser.parse_known_args()

    try:
        with log_context_duration(
            "phyloframe.legacy._alifestd_mark_sample_tips_polars",
            logging.info,
        ):
            _run_dataframe_cli(
                base_parser=parser,
                output_dataframe_op=functools.partial(
                    alifestd_mark_sample_tips_polars,
                    n_sample=args.n,
                    seed=args.seed,
                    mark_as=args.mark_as,
                ),
                overridden_arguments="ignore",  # seed is overridden
            )
    except NotImplementedError as e:
        logging.error(
            "- polars op not yet implemented, use pandas op CLI instead",
        )
        raise e
