import argparse
import contextlib
import functools
import gc
import logging
import os
import typing

import joinem
from joinem._dataframe_cli import _add_parser_base, _run_dataframe_cli
import opytional as opyt
import polars as pl
from tqdm import tqdm

from .._auxlib._RngStateContext import RngStateContext
from .._auxlib._add_bool_arg import add_bool_arg
from .._auxlib._begin_prod_logging import begin_prod_logging
from .._auxlib._format_cli_description import format_cli_description
from .._auxlib._get_phyloframe_version import get_phyloframe_version
from .._auxlib._log_context_duration import log_context_duration
from .._auxlib._log_memory_usage import log_memory_usage
from ._alifestd_calc_mrca_id_vector_asexual_polars import (
    alifestd_calc_mrca_id_vector_asexual_polars,
)
from ._alifestd_has_contiguous_ids_polars import (
    alifestd_has_contiguous_ids_polars,
)
from ._alifestd_is_topologically_sorted_polars import (
    alifestd_is_topologically_sorted_polars,
)
from ._alifestd_mark_leaves_polars import alifestd_mark_leaves_polars
from ._alifestd_mark_sample_tips_lineage_asexual import (
    _alifestd_downsample_tips_lineage_select_target_id,
)
from ._alifestd_mark_sample_tips_lineage_stratified_asexual import (
    _alifestd_downsample_tips_lineage_stratified_impl,
    _deprecate_n_tips,
)
from ._alifestd_try_add_ancestor_id_col_polars import (
    alifestd_try_add_ancestor_id_col_polars,
)


@_deprecate_n_tips
def alifestd_mark_sample_tips_lineage_stratified_polars(
    phylogeny_df: pl.DataFrame,
    n_sample: typing.Optional[int] = None,
    seed: typing.Optional[int] = None,
    *,
    criterion_delta: typing.Union[str, pl.Expr] = "origin_time",
    criterion_stratify: typing.Union[str, pl.Expr] = "origin_time",
    criterion_target: typing.Union[str, pl.Expr] = "origin_time",
    n_tips_per_stratum: int = 1,
    progress_wrap: typing.Callable = lambda x: x,
    mark_as: str = "alifestd_mark_sample_tips_lineage_stratified_polars",
) -> pl.DataFrame:
    """Mark leaves per stratified group, chosen by proximity to the
    lineage of a target leaf.

    Adds a boolean column ``mark_as`` indicating retained tips.

    Only supports asexual phylogenies.

    Parameters
    ----------
    phylogeny_df : polars.DataFrame or polars.LazyFrame
        The phylogeny as a dataframe in alife standard format.

        Must represent an asexual phylogeny.
    n_sample : int, optional
        Desired number of retained tips.  If ``None``, every distinct
        ``criterion_stratify`` value forms its own group.
    seed : int, optional
        Random seed for reproducible target-leaf selection.
    criterion_delta : str or polars.Expr, default "origin_time"
        Column name or polars expression used to compute the
        off-lineage delta for each leaf.
    criterion_stratify : str or polars.Expr, default "origin_time"
        Column name or polars expression used to stratify leaves into
        groups.
    criterion_target : str or polars.Expr, default "origin_time"
        Column name or polars expression used to select the target
        leaf.
    n_tips_per_stratum : int, default 1
        Number of tips to retain per stratified group.
    progress_wrap : Callable, optional
        Pass tqdm or equivalent to display a progress bar.
    mark_as : str, default "alifestd_mark_sample_tips_lineage_stratified_polars"
        Column name for the boolean mark.

    Raises
    ------
    NotImplementedError
        If `phylogeny_df` has no "ancestor_id" column or if ids are
        non-contiguous or not topologically sorted.
    ValueError
        If `criterion_delta`, `criterion_stratify`, or
        `criterion_target` is not a column in `phylogeny_df`.
    ValueError
        If ``n_sample`` is not ``None`` and ``n_tips_per_stratum``
        does not evenly divide ``n_sample``.

    Returns
    -------
    polars.DataFrame
        The phylogeny with an added boolean mark column.

    See Also
    --------
    alifestd_mark_sample_tips_lineage_stratified_asexual :
        Pandas-based implementation.
    """
    if n_sample is not None and n_sample % n_tips_per_stratum != 0:
        raise ValueError(
            f"n_tips_per_stratum={n_tips_per_stratum} does not evenly "
            f"divide n_sample={n_sample}",
        )

    logging.info(
        "- alifestd_mark_sample_tips_lineage_stratified_polars: "
        "adding ancestor_id col...",
    )
    phylogeny_df = alifestd_try_add_ancestor_id_col_polars(phylogeny_df)
    gc.collect()
    log_memory_usage(logging.info)

    logging.info(
        "- alifestd_mark_sample_tips_lineage_stratified_polars: "
        "collecting schema...",
    )
    schema_names = phylogeny_df.lazy().collect_schema().names()
    gc.collect()
    log_memory_usage(logging.info)

    if "ancestor_id" not in schema_names:
        raise NotImplementedError(
            "alifestd_mark_sample_tips_lineage_stratified_polars only "
            "supports asexual phylogenies.",
        )

    for criterion in (
        criterion_delta,
        criterion_stratify,
        criterion_target,
    ):
        if isinstance(criterion, str) and criterion not in schema_names:
            raise ValueError(
                f"criterion column {criterion!r} not found "
                f"in phylogeny_df",
            )

    if isinstance(criterion_delta, str):
        criterion_delta = pl.col(criterion_delta)
    if isinstance(criterion_stratify, str):
        criterion_stratify = pl.col(criterion_stratify)
    if isinstance(criterion_target, str):
        criterion_target = pl.col(criterion_target)

    logging.info(
        "- alifestd_mark_sample_tips_lineage_stratified_polars: "
        "checking empty...",
    )
    if phylogeny_df.lazy().limit(1).collect().is_empty():
        return phylogeny_df.with_columns(
            pl.lit(False).alias(mark_as),
        )
    gc.collect()
    log_memory_usage(logging.info)

    if not alifestd_has_contiguous_ids_polars(phylogeny_df):

        raise NotImplementedError(
            "non-contiguous ids not supported",
        )
    gc.collect()
    log_memory_usage(logging.info)

    if not alifestd_is_topologically_sorted_polars(phylogeny_df):

        raise NotImplementedError(
            "non-topologically-sorted data not supported",
        )
    gc.collect()
    log_memory_usage(logging.info)

    logging.info(
        "- alifestd_mark_sample_tips_lineage_stratified_polars: "
        "marking leaves...",
    )
    phylogeny_df = alifestd_mark_leaves_polars(phylogeny_df)
    gc.collect()
    log_memory_usage(logging.info)

    logging.info(
        "- alifestd_mark_sample_tips_lineage_stratified_polars: "
        "collecting is_leaf values...",
    )
    is_leaf = (
        phylogeny_df.lazy().select("is_leaf").collect().to_series().to_numpy()
    )
    gc.collect()
    log_memory_usage(logging.info)

    logging.info(
        "- alifestd_mark_sample_tips_lineage_stratified_polars: "
        "collecting criterion_target values...",
    )
    target_values = (
        phylogeny_df.lazy()
        .select(criterion_target)
        .collect()
        .to_series()
        .to_numpy()
    )
    gc.collect()
    log_memory_usage(logging.info)

    logging.info(
        "- alifestd_mark_sample_tips_lineage_stratified_polars: "
        "selecting target leaf...",
    )
    with opyt.apply_if_or_else(seed, RngStateContext, contextlib.nullcontext):
        target_id = _alifestd_downsample_tips_lineage_select_target_id(
            is_leaf, target_values
        )

    del target_values
    gc.collect()
    log_memory_usage(logging.info)

    logging.info(
        "- alifestd_mark_sample_tips_lineage_stratified_polars: "
        "collecting criterion_delta values...",
    )
    criterion_values = (
        phylogeny_df.lazy()
        .select(criterion_delta)
        .collect()
        .to_series()
        .to_numpy()
    )
    gc.collect()
    log_memory_usage(logging.info)

    logging.info(
        "- alifestd_mark_sample_tips_lineage_stratified_polars: "
        "collecting criterion_stratify values...",
    )
    stratify_values = (
        phylogeny_df.lazy()
        .select(criterion_stratify)
        .collect()
        .to_series()
        .to_numpy()
    )
    gc.collect()
    log_memory_usage(logging.info)

    logging.info(
        "- alifestd_mark_sample_tips_lineage_stratified_polars: "
        f"computing mrca vector for {target_id=}...",
    )
    mrca_vector = alifestd_calc_mrca_id_vector_asexual_polars(
        phylogeny_df, target_id=target_id, progress_wrap=progress_wrap
    )
    gc.collect()
    log_memory_usage(logging.info)

    logging.info(
        "- alifestd_mark_sample_tips_lineage_stratified_polars: "
        "computing is_marked...",
    )
    is_marked = _alifestd_downsample_tips_lineage_stratified_impl(
        is_leaf=is_leaf,
        criterion_values=criterion_values,
        stratify_values=stratify_values,
        mrca_vector=mrca_vector,
        n_sample=n_sample,
        n_tips_per_stratum=n_tips_per_stratum,
    )
    del criterion_values, is_leaf, mrca_vector, stratify_values
    gc.collect()
    log_memory_usage(logging.info)

    logging.info(
        "- alifestd_mark_sample_tips_lineage_stratified_polars: "
        "setting mark column...",
    )
    phylogeny_df = phylogeny_df.with_columns(
        pl.Series(name=mark_as, values=is_marked),
    )
    del is_marked
    gc.collect()
    log_memory_usage(logging.info)

    return phylogeny_df


_raw_description = f"""{os.path.basename(__file__)} | (phyloframe v{get_phyloframe_version()}/joinem v{joinem.__version__})

Mark one leaf per stratified group, chosen by proximity to the
lineage of a target leaf.

The target leaf is chosen as the leaf with the largest
`--criterion-target` value. For each leaf, the off-lineage delta is
the absolute difference between that leaf's `--criterion-delta` value
and the MRCA's `--criterion-delta` value (where the MRCA is of that
leaf and the target).
Leaves are grouped by their `--criterion-stratify` value. When `-n`
is given, stratified values are coarsened into `-n` groups by ranking
and integer division. Within each group, the leaf with the smallest
delta is marked.

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
phyloframe.legacy._alifestd_mark_sample_tips_lineage_stratified_asexual :
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
        dfcli_module="phyloframe.legacy._alifestd_mark_sample_tips_lineage_stratified_polars",
        dfcli_version=get_phyloframe_version(),
    )
    parser.add_argument(
        "-n",
        default=None,
        type=int,
        help="Number of stratified groups (default: one per distinct value).",
    )
    parser.add_argument(
        "--criterion-delta",
        default="origin_time",
        type=str,
        help="Column used to compute off-lineage delta (default: origin_time).",
    )
    parser.add_argument(
        "--criterion-stratify",
        default="origin_time",
        type=str,
        help="Column used to stratify leaves (default: origin_time).",
    )
    parser.add_argument(
        "--criterion-target",
        default="origin_time",
        type=str,
        help="Column used to select the target leaf (default: origin_time).",
    )
    parser.add_argument(
        "--n-tips-per-stratum",
        default=1,
        type=int,
        help="Number of tips per stratum (default: 1).",
    )
    parser.add_argument(
        "--seed",
        default=None,
        dest="seed",
        help="Integer seed for deterministic target-leaf selection.",
        type=int,
    )
    parser.add_argument(
        "--mark-as",
        default="alifestd_mark_sample_tips_lineage_stratified_polars",
        type=str,
        help="Column name for the boolean mark (default: alifestd_mark_sample_tips_lineage_stratified_polars).",
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
            "phyloframe.legacy._alifestd_mark_sample_tips_lineage_stratified_polars",
            logging.info,
        ):
            _run_dataframe_cli(
                base_parser=parser,
                output_dataframe_op=functools.partial(
                    alifestd_mark_sample_tips_lineage_stratified_polars,
                    n_sample=args.n,
                    seed=args.seed,
                    criterion_delta=args.criterion_delta,
                    criterion_stratify=args.criterion_stratify,
                    criterion_target=args.criterion_target,
                    n_tips_per_stratum=args.n_tips_per_stratum,
                    progress_wrap=tqdm,
                    mark_as=args.mark_as,
                ),
                overridden_arguments="ignore",  # seed is overridden
            )
    except NotImplementedError as e:
        logging.error(
            "- polars op not yet implemented, use pandas op CLI instead",
        )
        raise e
