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
from ._alifestd_mark_leaves_polars import alifestd_mark_leaves_polars


def alifestd_ultrametricize_polars(
    phylogeny_df: pl.DataFrame,
    *,
    method: typing.Literal["extend"] = "extend",
) -> pl.DataFrame:
    """Adjust tip `origin_time` values so all tips share the same time.

    With ``method="extend"``, each tip's ``origin_time`` is set to the
    maximum ``origin_time`` among tips. Internal node times are not
    modified.

    Empty phylogenies are returned unchanged. Must represent an asexual
    phylogeny (when ``is_leaf`` is not already present).

    See Also
    --------
    alifestd_ultrametricize :
        Pandas-based implementation.
    """
    if method != "extend":
        raise ValueError(
            f"alifestd_ultrametricize_polars: unknown method {method!r}",
        )

    schema_names = phylogeny_df.lazy().collect_schema().names()
    if "origin_time" not in schema_names:
        raise ValueError(
            "alifestd_ultrametricize_polars requires 'origin_time' column",
        )

    if phylogeny_df.lazy().limit(1).collect().is_empty():
        return phylogeny_df

    if "is_leaf" not in schema_names:
        phylogeny_df = alifestd_mark_leaves_polars(phylogeny_df)

    latest_origin_time = (
        phylogeny_df.lazy()
        .filter(pl.col("is_leaf"))
        .select(pl.col("origin_time").max())
        .collect()
        .item()
    )

    return phylogeny_df.with_columns(
        pl.when(pl.col("is_leaf"))
        .then(pl.lit(latest_origin_time))
        .otherwise(pl.col("origin_time"))
        .alias("origin_time"),
    )


_raw_description = f"""\
{os.path.basename(__file__)} | \
(phyloframe v{get_phyloframe_version()}/joinem v{joinem.__version__})

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
phyloframe.legacy._alifestd_ultrametricize :
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
        dfcli_module="phyloframe.legacy._alifestd_ultrametricize_polars",
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

    try:
        with log_context_duration(
            "phyloframe.legacy._alifestd_ultrametricize_polars",
            logging.info,
        ):
            _run_dataframe_cli(
                base_parser=parser,
                output_dataframe_op=functools.partial(
                    alifestd_ultrametricize_polars, method=args.method
                ),
            )
    except NotImplementedError as e:
        logging.error(
            "- polars op not yet implemented, use pandas op CLI instead",
        )
        raise e
