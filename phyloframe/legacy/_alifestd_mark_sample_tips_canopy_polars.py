import argparse
import functools
import gc
import logging
import os
import typing

import joinem
from joinem._dataframe_cli import _add_parser_base, _run_dataframe_cli
import polars as pl

from .._auxlib._add_bool_arg import add_bool_arg
from .._auxlib._begin_prod_logging import begin_prod_logging
from .._auxlib._format_cli_description import format_cli_description
from .._auxlib._get_phyloframe_version import get_phyloframe_version
from .._auxlib._log_context_duration import log_context_duration
from .._auxlib._log_memory_usage import log_memory_usage
from ._alifestd_downsample_tips_canopy_asexual import (
    _deprecate_num_tips,
)
from ._alifestd_mark_leaves_polars import alifestd_mark_leaves_polars


@_deprecate_num_tips
def alifestd_mark_sample_tips_canopy_polars(
    phylogeny_df: pl.DataFrame,
    n_sample: typing.Optional[int] = None,
    criterion: typing.Union[str, pl.Expr] = "origin_time",
    *,
    mark_as: str = "alifestd_mark_sample_tips_canopy_polars",
) -> pl.DataFrame:
    """Mark the `n_sample` leaves with the largest `criterion` values.

    Adds a boolean column ``mark_as`` indicating retained tips.

    If `n_sample` is ``None``, it defaults to the number of leaves that
    share the maximum value of the `criterion` column. If `n_sample` is
    greater than or equal to the number of leaves in the phylogeny, all
    leaves are marked.  Ties are broken arbitrarily.

    Only supports asexual phylogenies.

    Parameters
    ----------
    phylogeny_df : polars.DataFrame
        The phylogeny as a dataframe in alife standard format.

        Must represent an asexual phylogeny.
    n_sample : int, optional
        Number of tips to mark. If ``None``, defaults to the count of
        leaves with the maximum `criterion` value.
    criterion : str or polars.Expr, default "origin_time"
        Column name or polars expression used to rank leaves. The
        `n_sample` leaves with the largest values are marked. Ties are
        broken arbitrarily.
    mark_as : str, default "alifestd_mark_sample_tips_canopy_polars"
        Column name for the boolean mark.

    Raises
    ------
    NotImplementedError
        If `phylogeny_df` has no "ancestor_id" column.
    ValueError
        If `criterion` is not a column in `phylogeny_df`.

    Returns
    -------
    polars.DataFrame
        The phylogeny with an added boolean mark column.

    See Also
    --------
    alifestd_mark_sample_tips_canopy_asexual :
        Pandas-based implementation.
    """
    logging.info(
        "- alifestd_mark_sample_tips_canopy_polars: collecting schema...",
    )
    schema_names = phylogeny_df.lazy().collect_schema().names()
    gc.collect()
    log_memory_usage(logging.info)

    if isinstance(criterion, str):
        if criterion not in schema_names:
            raise ValueError(
                f"criterion column {criterion!r} not found "
                f"in phylogeny_df",
            )
        criterion = pl.col(criterion)

    if "ancestor_id" not in schema_names:
        raise NotImplementedError("ancestor_id column required")

    logging.info(
        "- alifestd_mark_sample_tips_canopy_polars: checking empty...",
    )
    if phylogeny_df.lazy().limit(1).collect().is_empty():
        return phylogeny_df.with_columns(
            pl.lit(False).alias(mark_as),
        )

    logging.info(
        "- alifestd_mark_sample_tips_canopy_polars: finding leaf ids...",
    )
    phylogeny_df = alifestd_mark_leaves_polars(phylogeny_df)
    gc.collect()
    log_memory_usage(logging.info)

    logging.info(
        "- alifestd_mark_sample_tips_canopy_polars: selecting top leaf_ids...",
    )
    leaves_lazy = phylogeny_df.lazy().filter(pl.col("is_leaf"))
    if n_sample is None:
        max_val = leaves_lazy.select(criterion.max()).collect().item()
        n_sample = (
            leaves_lazy.filter(criterion == max_val)
            .select(pl.len())
            .collect()
            .item()
        )
        gc.collect()
        log_memory_usage(logging.info)

    logging.info(
        "- alifestd_mark_sample_tips_canopy_polars: counting leaves...",
    )
    total_leaves = leaves_lazy.select(pl.len()).collect().item()
    logging.info(
        f"- alifestd_mark_sample_tips_canopy_polars: {total_leaves=}...",
    )

    if n_sample >= total_leaves:
        logging.info(
            "- alifestd_mark_sample_tips_canopy_polars: taking all...",
        )
        leaf_ids = leaves_lazy.select(pl.col("id")).collect().to_series()
    else:  # split case to prevent extreme top_k crash where n_sample is high
        logging.info(
            "- alifestd_mark_sample_tips_canopy_polars: taking top k...",
        )
        leaf_ids = (
            leaves_lazy.top_k(n_sample, by=criterion)
            .select(pl.col("id"))
            .collect()
            .to_series()
        )
    gc.collect()
    log_memory_usage(logging.info)

    logging.info(
        "- alifestd_mark_sample_tips_canopy_polars: setting mark column...",
    )
    phylogeny_df = phylogeny_df.with_columns(
        pl.col("id").is_in(leaf_ids).alias(mark_as),
    )
    del leaf_ids
    gc.collect()
    log_memory_usage(logging.info)

    return phylogeny_df


_raw_description = f"""{os.path.basename(__file__)} | (phyloframe v{get_phyloframe_version()}/joinem v{joinem.__version__})

Mark the `-n` leaves with the largest `--criterion` values.

If `-n` is greater than or equal to the number of leaves in the phylogeny, all leaves are marked. Ties are broken arbitrarily.

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
phyloframe.legacy._alifestd_mark_sample_tips_canopy_asexual :
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
        dfcli_module="phyloframe.legacy._alifestd_mark_sample_tips_canopy_polars",
        dfcli_version=get_phyloframe_version(),
    )
    parser.add_argument(
        "-n",
        default=None,
        type=int,
        help="Number of tips to mark. If omitted, defaults to the count of leaves with the maximum criterion value.",
    )
    parser.add_argument(
        "--criterion",
        default="origin_time",
        type=str,
        help="Column name used to rank leaves; ties broken arbitrarily (default: origin_time).",
    )
    parser.add_argument(
        "--mark-as",
        default="alifestd_mark_sample_tips_canopy_polars",
        type=str,
        help="Column name for the boolean mark (default: alifestd_mark_sample_tips_canopy_polars).",
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
            "phyloframe.legacy._alifestd_mark_sample_tips_canopy_polars",
            logging.info,
        ):
            _run_dataframe_cli(
                base_parser=parser,
                output_dataframe_op=functools.partial(
                    alifestd_mark_sample_tips_canopy_polars,
                    n_sample=args.n,
                    criterion=args.criterion,
                    mark_as=args.mark_as,
                ),
            )
    except NotImplementedError as e:
        logging.error(
            "- polars op not yet implemented, use pandas op CLI instead",
        )
        raise e
