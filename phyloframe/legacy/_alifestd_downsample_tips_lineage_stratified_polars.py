import argparse
import functools
import logging
import os
import typing

import joinem
from joinem._dataframe_cli import _add_parser_base, _run_dataframe_cli
import polars as pl
from tqdm import tqdm

from .._auxlib._add_bool_arg import add_bool_arg
from .._auxlib._begin_prod_logging import begin_prod_logging
from .._auxlib._format_cli_description import format_cli_description
from .._auxlib._get_phyloframe_version import get_phyloframe_version
from .._auxlib._log_context_duration import log_context_duration
from ._alifestd_mark_sample_tips_lineage_stratified_asexual import (
    _deprecate_n_tips,
)
from ._alifestd_mark_sample_tips_lineage_stratified_polars import (
    alifestd_mark_sample_tips_lineage_stratified_polars,
)
from ._alifestd_prune_extinct_lineages_polars import (
    alifestd_prune_extinct_lineages_polars,
)
from ._alifestd_topological_sensitivity_warned_polars import (
    alifestd_topological_sensitivity_warned_polars,
)


@_deprecate_n_tips
@alifestd_topological_sensitivity_warned_polars(
    insert=False,
    delete=True,
    update=False,
)
def alifestd_downsample_tips_lineage_stratified_polars(
    phylogeny_df: pl.DataFrame,
    n_downsample: typing.Optional[int] = None,
    seed: typing.Optional[int] = None,
    *,
    criterion_delta: typing.Union[str, pl.Expr] = "origin_time",
    criterion_stratify: typing.Union[str, pl.Expr] = "origin_time",
    criterion_target: typing.Union[str, pl.Expr] = "origin_time",
    n_tips_per_stratum: int = 1,
    progress_wrap: typing.Callable = lambda x: x,
) -> pl.DataFrame:
    """Retain leaves per stratified group, chosen by proximity to the
    lineage of a target leaf.

    Selects a target leaf as the leaf with the largest `criterion_target`
    value (ties broken randomly). For each non-target leaf, the most
    recent common ancestor (MRCA) of that leaf and the target leaf is
    identified, and the "off-lineage delta" is computed as the absolute
    difference between that leaf's `criterion_delta` value and the
    MRCA's `criterion_delta` value.

    Leaves are grouped by their `criterion_stratify` value. When
    `n_downsample` is an integer, stratified values are coarsened by
    ranking and integer-dividing to form exactly
    ``n_downsample // n_tips_per_stratum`` groups. When `n_downsample`
    is ``None``, each distinct stratified value forms its own group
    (without ranking). Within each group, the ``n_tips_per_stratum``
    leaves with the smallest off-lineage delta are retained.

    Only supports asexual phylogenies.

    Parameters
    ----------
    phylogeny_df : polars.DataFrame or polars.LazyFrame
        The phylogeny as a dataframe in alife standard format.

        Must represent an asexual phylogeny.
    n_downsample : int, optional
        Desired number of retained tips.  If ``None``, every distinct
        ``criterion_stratify`` value forms its own group.
    seed : int, optional
        Random seed for reproducible target-leaf selection when there are
        ties in `criterion_target`.
    criterion_delta : str or polars.Expr, default "origin_time"
        Column name or polars expression used to compute the
        off-lineage delta for each leaf. The delta is the absolute
        difference between a leaf's value and its MRCA's value.
    criterion_stratify : str or polars.Expr, default "origin_time"
        Column name or polars expression used to stratify leaves into
        groups.
    criterion_target : str or polars.Expr, default "origin_time"
        Column name or polars expression used to select the target
        leaf. The leaf with the largest value is chosen as the target.
        Note that ties are broken by random sample, allowing a seed to
        be provided.
    n_tips_per_stratum : int, default 1
        Number of tips to retain per stratified group.  Must evenly
        divide ``n_downsample`` when ``n_downsample`` is not ``None``.
    progress_wrap : Callable, optional
        Pass tqdm or equivalent to display a progress bar.

    Raises
    ------
    NotImplementedError
        If `phylogeny_df` has no "ancestor_id" column or if ids are
        non-contiguous or not topologically sorted.
    ValueError
        If `criterion_delta`, `criterion_stratify`, or
        `criterion_target` is not a column in `phylogeny_df`.
    ValueError
        If ``n_downsample`` is not ``None`` and ``n_tips_per_stratum``
        does not evenly divide ``n_downsample``.

    Returns
    -------
    polars.DataFrame
        The pruned phylogeny in alife standard format.

    See Also
    --------
    alifestd_downsample_tips_lineage_stratified_asexual :
        Pandas-based implementation.
    """
    phylogeny_df = alifestd_mark_sample_tips_lineage_stratified_polars(
        phylogeny_df,
        n_sample=n_downsample,
        seed=seed,
        criterion_delta=criterion_delta,
        criterion_stratify=criterion_stratify,
        criterion_target=criterion_target,
        n_tips_per_stratum=n_tips_per_stratum,
        progress_wrap=progress_wrap,
        mark_as="extant",
    )

    if phylogeny_df.lazy().limit(1).collect().is_empty():
        return phylogeny_df.drop("extant")

    logging.info(
        "- alifestd_downsample_tips_lineage_stratified_polars: pruning...",
    )
    return alifestd_prune_extinct_lineages_polars(phylogeny_df).drop(
        "extant",
    )


_raw_description = f"""{os.path.basename(__file__)} | (phyloframe v{get_phyloframe_version()}/joinem v{joinem.__version__})

Retain one leaf per stratified group, chosen by proximity to the
lineage of a target leaf.

The target leaf is chosen as the leaf with the largest
`--criterion-target` value. For each leaf, the off-lineage delta is
the absolute difference between that leaf's `--criterion-delta` value
and the MRCA's `--criterion-delta` value (where the MRCA is of that
leaf and the target).
Leaves are grouped by their `--criterion-stratify` value. When `-n`
is given, stratified values are coarsened into `-n` groups by ranking
and integer division. Within each group, the leaf with the smallest
delta is retained.

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
phyloframe.legacy._alifestd_downsample_tips_lineage_stratified_asexual :
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
        dfcli_module="phyloframe.legacy._alifestd_downsample_tips_lineage_stratified_polars",
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
            "phyloframe.legacy._alifestd_downsample_tips_lineage_stratified_polars",
            logging.info,
        ):
            _run_dataframe_cli(
                base_parser=parser,
                output_dataframe_op=functools.partial(
                    alifestd_downsample_tips_lineage_stratified_polars,
                    n_downsample=args.n,
                    seed=args.seed,
                    criterion_delta=args.criterion_delta,
                    criterion_stratify=args.criterion_stratify,
                    criterion_target=args.criterion_target,
                    n_tips_per_stratum=args.n_tips_per_stratum,
                    progress_wrap=tqdm,
                    ignore_topological_sensitivity=args.ignore_topological_sensitivity,
                    drop_topological_sensitivity=args.drop_topological_sensitivity,
                ),
                overridden_arguments="ignore",  # seed is overridden
            )
    except NotImplementedError as e:
        logging.error(
            "- polars op not yet implemented, use pandas op CLI instead",
        )
        raise e
