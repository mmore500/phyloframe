import argparse
import contextlib
import functools
import logging
import os
import typing

import joinem
from joinem._dataframe_cli import _add_parser_base, _run_dataframe_cli
import opytional as opyt
import polars as pl

from .._auxlib._RngStateContext import RngStateContext
from .._auxlib._begin_prod_logging import begin_prod_logging
from .._auxlib._format_cli_description import format_cli_description
from .._auxlib._get_phyloframe_version import get_phyloframe_version
from .._auxlib._log_context_duration import log_context_duration
from ._alifestd_assign_contiguous_ids_polars import (
    alifestd_assign_contiguous_ids_polars,
)
from ._alifestd_make_ancestor_list_col_polars import (
    alifestd_make_ancestor_list_col_polars,
)
from ._alifestd_shuffle_working_format import (
    _alifestd_shuffle_working_format_permutation,
)
from ._alifestd_topological_sort_polars import (
    alifestd_topological_sort_polars,
)
from ._alifestd_try_add_ancestor_id_col_polars import (
    alifestd_try_add_ancestor_id_col_polars,
)


def alifestd_shuffle_working_format_polars(
    phylogeny_df: pl.DataFrame,
    *,
    seed: typing.Optional[int] = None,
) -> pl.DataFrame:
    """Randomly reorder rows, then restore working format.

    The returned phylogeny dataframe represents the same phylogeny, but with
    rows in a random topologically sorted order and ids reassigned to be
    contiguous (i.e., organisms' ids correspond to row number). Useful for
    testing that operations do not depend on row order or id assignment.

    Track taxa across the shuffle with a label column, as `id` values are
    reassigned.

    Parameters
    ----------
    phylogeny_df : polars.DataFrame or polars.LazyFrame
        The phylogeny as a dataframe in alife standard format.

        Must represent an asexual phylogeny. If present, `ancestor_list` is
        regenerated to reflect reassigned ids.
    seed : int, optional
        Random seed for shuffling.

    Returns
    -------
    polars.DataFrame
        The shuffled phylogeny, in working format.

    See Also
    --------
    alifestd_shuffle_working_format :
        Pandas-based implementation.
    """
    logging.info(
        "- alifestd_shuffle_working_format_polars: adding ancestor_id col...",
    )
    phylogeny_df = alifestd_try_add_ancestor_id_col_polars(phylogeny_df)
    phylogeny_df = phylogeny_df.lazy().collect()

    if "ancestor_id" not in phylogeny_df.columns:
        raise NotImplementedError("sexual phylogenies not supported")

    if phylogeny_df.is_empty():
        return phylogeny_df

    has_ancestor_list = "ancestor_list" in phylogeny_df.columns
    if has_ancestor_list:  # regenerated below, after ids are reassigned
        phylogeny_df = phylogeny_df.drop("ancestor_list")

    logging.info("- alifestd_shuffle_working_format_polars: shuffling...")
    with opyt.apply_if_or_else(seed, RngStateContext, contextlib.nullcontext):
        permutation = _alifestd_shuffle_working_format_permutation(
            len(phylogeny_df),
        )
    phylogeny_df = phylogeny_df.select(pl.all().gather(permutation))

    # relabel ids to shuffled row positions, so that topological sort, which
    # orders by id, inherits shuffled order...
    phylogeny_df = alifestd_assign_contiguous_ids_polars(phylogeny_df)
    logging.info("- alifestd_shuffle_working_format_polars: sorting...")
    phylogeny_df = alifestd_topological_sort_polars(phylogeny_df)
    # ... then relabel ids to sorted row positions
    phylogeny_df = alifestd_assign_contiguous_ids_polars(phylogeny_df)
    phylogeny_df = phylogeny_df.lazy().collect()

    if has_ancestor_list:
        phylogeny_df = phylogeny_df.with_columns(
            ancestor_list=alifestd_make_ancestor_list_col_polars(
                phylogeny_df["id"], phylogeny_df["ancestor_id"]
            ),
        )

    return phylogeny_df


_raw_description = f"""{os.path.basename(__file__)} | (phyloframe v{get_phyloframe_version()}/joinem v{joinem.__version__})

Randomly reorder rows, then restore working format (topologically sorted, contiguous ids).

Data is assumed to be in alife standard format.

Additional Notes
================
- Requires asexual phylogeny.

- Use `--seed` for reproducible shuffling.

- Use `--eager-read` if modifying data file inplace.

- This CLI entrypoint is experimental and may be subject to change.

See Also
========
phyloframe.legacy._alifestd_shuffle_working_format :
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
        dfcli_module="phyloframe.legacy._alifestd_shuffle_working_format_polars",
        dfcli_version=get_phyloframe_version(),
    )
    parser.add_argument(
        "--seed",
        default=None,
        type=int,
        help="random seed (default: None)",
    )
    return parser


if __name__ == "__main__":
    begin_prod_logging()

    parser = _create_parser()
    args, __ = parser.parse_known_args()

    try:
        with log_context_duration(
            "phyloframe.legacy._alifestd_shuffle_working_format_polars",
            logging.info,
        ):
            _run_dataframe_cli(
                base_parser=parser,
                output_dataframe_op=functools.partial(
                    alifestd_shuffle_working_format_polars,
                    seed=args.seed,
                ),
                overridden_arguments="ignore",  # seed is overridden
            )
    except NotImplementedError as e:
        logging.error(
            "- polars op not yet implemented, use pandas op CLI instead",
        )
        raise e
