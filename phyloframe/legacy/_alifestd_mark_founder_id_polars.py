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


def alifestd_mark_founder_id_polars(
    phylogeny_df: pl.DataFrame,
    *,
    founder_mask: typing.Union[str, pl.Expr],
    mark_as: str = "founder_id",
) -> pl.DataFrame:
    """Add column `founder_id`, containing the `id` of the oldest taxon along
    each entry's lineage (inclusive) marked by `founder_mask`.

    Entries with no marked taxon along their lineage are marked as their own
    founder.

    The output column name can be changed via the ``mark_as`` parameter.

    Only supports asexual phylogenies.

    Parameters
    ----------
    phylogeny_df : polars.DataFrame
        The phylogeny as a dataframe in alife standard format.

        Must represent an asexual phylogeny with contiguous,
        topologically sorted ids.
    founder_mask : str or polars.Expr
        Column name or boolean expression marking founder taxa.
    mark_as : str, default "founder_id"
        Column name for the mark.

    Raises
    ------
    NotImplementedError
        If `phylogeny_df` has non-contiguous ids or is not topologically
        sorted.

    Returns
    -------
    polars.DataFrame
        The phylogeny with an added mark column.

    See Also
    --------
    alifestd_mark_root_id_polars :
        Marks ids of entries' ultimate ancestors.
    """
    if isinstance(founder_mask, str):
        founder_mask = pl.col(founder_mask)

    logging.info(
        "- alifestd_mark_founder_id_polars: finding oldest founders...",
    )
    # non-founders get sentinel len; ids increase from ancestor to descendant
    phylogeny_df = alifestd_mark_lineage_cummin_polars(
        phylogeny_df,
        pl.when(founder_mask.cast(pl.Boolean))
        .then(pl.int_range(pl.len(), dtype=pl.Int64))
        .otherwise(pl.len().cast(pl.Int64)),
        mark_as=mark_as,
    )
    return phylogeny_df.with_columns(
        pl.when(pl.col(mark_as) < pl.len())
        .then(pl.col(mark_as))
        .otherwise(pl.col("id"))
        .alias(mark_as),
    )


_raw_description = f"""{os.path.basename(__file__)} | (phyloframe v{get_phyloframe_version()}/joinem v{joinem.__version__})

Add column `founder_id`, containing the `id` of the oldest taxon along each
entry's lineage (inclusive) marked by `--founder-mask`.

Entries with no marked taxon along their lineage are their own founder.

Data is assumed to be in alife standard format.
Only supports asexual phylogenies.

Additional Notes
================
- Requires 'ancestor_id' column to be present in input DataFrame.
Otherwise, no action is taken.

- Use `--with-column` to add a founder mask column, e.g.,
`--with-column '(pl.col("depth") == 300).alias("is_founder")'`.

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
        dfcli_module="phyloframe.legacy._alifestd_mark_founder_id_polars",
        dfcli_version=get_phyloframe_version(),
    )
    parser.add_argument(
        "--founder-mask",
        default=None,
        type=str,
        help="Name of boolean column marking founder taxa.",
    )
    parser.add_argument(
        "--mark-as",
        default="founder_id",
        type=str,
        help="Column name for the mark (default: founder_id).",
    )
    return parser


if __name__ == "__main__":
    begin_prod_logging()

    parser = _create_parser()
    args, __ = parser.parse_known_args()

    with log_context_duration(
        "phyloframe.legacy._alifestd_mark_founder_id_polars",
        logging.info,
    ):
        _run_dataframe_cli(
            base_parser=parser,
            output_dataframe_op=functools.partial(
                alifestd_mark_founder_id_polars,
                founder_mask=args.founder_mask,
                mark_as=args.mark_as,
            ),
        )
