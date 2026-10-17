import argparse
import functools
import gc
import logging
import os
import typing

import joinem
from joinem._dataframe_cli import _add_parser_base, _run_dataframe_cli
import polars as pl

from .._auxlib._begin_prod_logging import begin_prod_logging
from .._auxlib._format_cli_description import format_cli_description
from .._auxlib._get_phyloframe_version import get_phyloframe_version
from .._auxlib._log_context_duration import log_context_duration
from .._auxlib._log_memory_usage import log_memory_usage
from ._alifestd_mask_monomorphic_clades_polars import (
    alifestd_mask_monomorphic_clades_polars,
)


def alifestd_mark_monomorphic_clades_polars(
    phylogeny_df: pl.DataFrame,
    trait_by: typing.Union[
        str,
        pl.Expr,
        typing.Sequence[typing.Union[str, pl.Expr]],
    ],
    *,
    trait_mask: typing.Optional[typing.Union[str, pl.Expr]] = None,
    mark_as: str = "alifestd_mark_monomorphic_clades_polars",
) -> pl.DataFrame:
    """Mark "monomorphic" clades where all members with a defined trait share
    identical values in `trait_by` column(s) or expression(s).

    Clades containing no members with a defined trait value are considered
    monomorphic. All leaf nodes are considered monomorphic. Null values
    compare equal to one another.

    Only supports asexual phylogenies.

    Parameters
    ----------
    phylogeny_df : polars.DataFrame
        The phylogeny as a dataframe in alife standard format.

        Must represent an asexual phylogeny with contiguous,
        topologically sorted ids.
    trait_by : str, polars.Expr, or sequence of str or polars.Expr
        Column name(s) or expression(s) whose combined values determine
        monomorphism.
    trait_mask : str or polars.Expr, optional
        Column name or boolean expression marking taxa with a defined trait
        value. Values of unmarked taxa are ignored. For example, pass
        "is_leaf" to consider only leaf values (see
        `alifestd_mark_leaves_polars`). If None, all taxa are considered.
    mark_as : str, default "alifestd_mark_monomorphic_clades_polars"
        Column name for the boolean mark.

    Raises
    ------
    NotImplementedError
        If `phylogeny_df` has non-contiguous ids or is not topologically
        sorted.
    ValueError
        If a column named in `trait_by` or `trait_mask` is not in `phylogeny_df`.

    Returns
    -------
    polars.DataFrame
        The phylogeny with an added boolean mark column.

    See Also
    --------
    alifestd_mask_monomorphic_clades_polars :
        Marks monomorphic clades given trait mask and value arrays.
    """
    if isinstance(trait_by, (str, pl.Expr)):
        trait_by = [trait_by]
    else:
        trait_by = [*trait_by]

    logging.info(
        "- alifestd_mark_monomorphic_clades_polars: collecting schema...",
    )
    schema_names = phylogeny_df.lazy().collect_schema().names()
    for col in (trait_mask, *trait_by):
        if isinstance(col, str) and col not in schema_names:
            raise ValueError(f"column {col!r} not found in phylogeny_df")

    if trait_mask is None:
        trait_mask = pl.repeat(True, pl.len())
    elif isinstance(trait_mask, str):
        trait_mask = pl.col(trait_mask)

    logging.info(
        "- alifestd_mark_monomorphic_clades_polars: masking clades...",
    )
    phylogeny_df = alifestd_mask_monomorphic_clades_polars(
        phylogeny_df,
        mark_as=mark_as,
        trait_mask=phylogeny_df.lazy()
        .select(trait_mask.cast(pl.Boolean))
        .collect()
        .to_series()
        .to_numpy(),
        trait_values=phylogeny_df.lazy()
        .select(
            trait_by[0]
            if len(trait_by) == 1
            else pl.struct(trait_by).rank("dense")
        )
        .collect()
        .to_series()
        .to_numpy(),
    )
    gc.collect()
    log_memory_usage(logging.info)

    return phylogeny_df


_raw_description = f"""{os.path.basename(__file__)} | (phyloframe v{get_phyloframe_version()}/joinem v{joinem.__version__})

Mark "monomorphic" clades where all members with a defined trait share
identical values in `--trait-by` column(s).

Clades containing no members with a defined trait value are considered
monomorphic. All leaf nodes are considered monomorphic.

Data is assumed to be in alife standard format.
Only supports asexual phylogenies.

Additional Notes
================
- Requires 'ancestor_id' column to be present in input DataFrame.
Otherwise, no action is taken.

- Use `--trait-mask` to consider only values of taxa marked by a boolean
column, e.g., `--trait-mask is_leaf` to consider only leaf values.

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
        dfcli_module="phyloframe.legacy._alifestd_mark_monomorphic_clades_polars",
        dfcli_version=get_phyloframe_version(),
    )
    parser.add_argument(
        "--trait-by",
        action="append",
        dest="trait_by",
        type=str,
        default=[],
        help=(
            "Name of column whose values determine monomorphism. "
            "Specify multiple columns by using this flag multiple times."
        ),
    )
    parser.add_argument(
        "--trait-mask",
        default=None,
        type=str,
        help=(
            "Name of boolean column marking taxa whose values are "
            "considered (default: all taxa)."
        ),
    )
    parser.add_argument(
        "--mark-as",
        default="alifestd_mark_monomorphic_clades_polars",
        type=str,
        help=(
            "Column name for the boolean mark "
            "(default: alifestd_mark_monomorphic_clades_polars)."
        ),
    )
    return parser


if __name__ == "__main__":
    begin_prod_logging()

    parser = _create_parser()
    args, __ = parser.parse_known_args()

    with log_context_duration(
        "phyloframe.legacy._alifestd_mark_monomorphic_clades_polars",
        logging.info,
    ):
        _run_dataframe_cli(
            base_parser=parser,
            output_dataframe_op=functools.partial(
                alifestd_mark_monomorphic_clades_polars,
                trait_by=args.trait_by,
                trait_mask=args.trait_mask,
                mark_as=args.mark_as,
            ),
        )
