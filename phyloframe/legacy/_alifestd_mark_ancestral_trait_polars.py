import argparse
import functools
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
from ._alifestd_mark_lineage_cummin_polars import (
    alifestd_mark_lineage_cummin_polars,
)


def alifestd_mark_ancestral_trait_polars(
    phylogeny_df: pl.DataFrame,
    *,
    mark_as: str = "ancestral_trait",
    trait_default: typing.Union[str, pl.Expr],
    trait_mask: typing.Union[str, pl.Expr],
    trait_values: typing.Union[str, pl.Expr],
) -> pl.DataFrame:
    """Mark each taxon with the trait value of the oldest taxon along its
    lineage (inclusive) that has a defined trait value, or `trait_default`
    if no such taxon exists.

    Adds a column ``mark_as``.

    Only supports asexual phylogenies.

    Parameters
    ----------
    phylogeny_df : polars.DataFrame
        The phylogeny as a dataframe in alife standard format.

        Must represent an asexual phylogeny with contiguous,
        topologically sorted ids.
    mark_as : str, default "ancestral_trait"
        Column name for the mark.
    trait_default : str or polars.Expr
        Column name or expression giving values for taxa with no defined
        trait value along their lineage. Pass a scalar as `pl.lit(...)`.
    trait_mask : str or polars.Expr
        Column name or boolean expression marking taxa that have a defined
        trait value.
    trait_values : str or polars.Expr
        Column name or expression giving trait values.

    Raises
    ------
    NotImplementedError
        If `phylogeny_df` has non-contiguous ids or is not topologically
        sorted.
    ValueError
        If a column named by `trait_default`, `trait_mask`, or `trait_values`
        is not in `phylogeny_df`.

    Returns
    -------
    polars.DataFrame
        The phylogeny with an added mark column.

    See Also
    --------
    alifestd_mark_lineage_cummin_polars :
        Marks minimum of values along each lineage.
    """
    logging.info(
        "- alifestd_mark_ancestral_trait_polars: collecting schema...",
    )
    schema_names = phylogeny_df.lazy().collect_schema().names()
    trait_default, trait_mask, trait_values = (
        pl.col(col) if isinstance(col, str) else col
        for col in (trait_default, trait_mask, trait_values)
    )
    for col in (trait_default, trait_mask, trait_values):
        for name in col.meta.root_names():
            if name not in schema_names:
                raise ValueError(f"column {name!r} not found in phylogeny_df")

    logging.info(
        "- alifestd_mark_ancestral_trait_polars: finding oldest marked...",
    )
    # unmarked taxa get sentinel len; ids increase from ancestor to descendant
    oldest_col = "alifestd_mark_ancestral_trait_oldest"
    phylogeny_df = alifestd_mark_lineage_cummin_polars(
        phylogeny_df,
        pl.when(trait_mask.cast(pl.Boolean))
        .then(pl.int_range(pl.len(), dtype=pl.Int64))
        .otherwise(pl.len().cast(pl.Int64)),
        mark_as=oldest_col,
    )

    logging.info(
        "- alifestd_mark_ancestral_trait_polars: assigning trait values...",
    )
    oldest = pl.col(oldest_col)
    return phylogeny_df.with_columns(
        pl.when(oldest < pl.len())
        .then(
            trait_values.gather(
                oldest.clip(upper_bound=pl.len().cast(pl.Int64) - 1)
            )
        )
        .otherwise(trait_default)
        .alias(mark_as),
    ).drop(oldest_col)


_raw_description = f"""{os.path.basename(__file__)} | (phyloframe v{get_phyloframe_version()}/joinem v{joinem.__version__})

Mark each taxon with the trait value of the oldest taxon along its lineage
(inclusive) that has a defined trait value, or `--trait-default` if no such
taxon exists.

Data is assumed to be in alife standard format.
Only supports asexual phylogenies.

Additional Notes
================
- Requires 'ancestor_id' column to be present in input DataFrame.
Otherwise, no action is taken.

- `--trait-default`, `--trait-mask`, and `--trait-values` name columns. Use
`--with-column` to add a column with a scalar default, e.g.,
`--with-column 'pl.lit(-1).alias("default")' --trait-default default`.

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
        dfcli_module="phyloframe.legacy._alifestd_mark_ancestral_trait_polars",
        dfcli_version=get_phyloframe_version(),
    )
    parser.add_argument(
        "--mark-as",
        default="ancestral_trait",
        type=str,
        help="Column name for the mark (default: ancestral_trait).",
    )
    parser.add_argument(
        "--trait-default",
        default=None,
        type=str,
        help="Name of column with values for taxa without a marked ancestor.",
    )
    parser.add_argument(
        "--trait-mask",
        default=None,
        type=str,
        help="Name of boolean column marking taxa with defined trait values.",
    )
    parser.add_argument(
        "--trait-values",
        default=None,
        type=str,
        help="Name of column with trait values.",
    )
    return parser


if __name__ == "__main__":
    begin_prod_logging()

    parser = _create_parser()
    args, __ = parser.parse_known_args()

    try:
        with log_context_duration(
            "phyloframe.legacy._alifestd_mark_ancestral_trait_polars",
            logging.info,
        ):
            _run_dataframe_cli(
                base_parser=parser,
                output_dataframe_op=functools.partial(
                    alifestd_mark_ancestral_trait_polars,
                    mark_as=args.mark_as,
                    trait_default=args.trait_default,
                    trait_mask=args.trait_mask,
                    trait_values=args.trait_values,
                ),
            )
    except NotImplementedError as e:
        logging.error(
            "- polars op not yet implemented, use pandas op CLI instead",
        )
        raise e
