import argparse
import functools
import logging
import os
import typing

import joinem
from joinem._dataframe_cli import _add_parser_base, _run_dataframe_cli
import numpy as np
import polars as pl

from .._auxlib._begin_prod_logging import begin_prod_logging
from .._auxlib._format_cli_description import format_cli_description
from .._auxlib._get_phyloframe_version import get_phyloframe_version
from .._auxlib._log_context_duration import log_context_duration
from ._alifestd_assign_contiguous_ids_polars import (
    alifestd_assign_contiguous_ids_polars,
)
from ._alifestd_collapse_unifurcations_polars import (
    alifestd_collapse_unifurcations_polars,
)
from ._alifestd_estimate_trait_gain_selection_asexual import (
    _tgs_impl,
)
from ._alifestd_has_contiguous_ids_polars import (
    alifestd_has_contiguous_ids_polars,
)
from ._alifestd_is_topologically_sorted_polars import (
    alifestd_is_topologically_sorted_polars,
)
from ._alifestd_mask_leaf_trait_inherited_polars import (
    _alifestd_mask_leaf_trait_inherited_polars_fill_null,
)
from ._alifestd_try_add_ancestor_id_col_polars import (
    alifestd_try_add_ancestor_id_col_polars,
)


def alifestd_estimate_trait_gain_selection_polars(
    phylogeny_df: pl.DataFrame,
    *,
    trait_masks: typing.Sequence[typing.Union[str, pl.Expr]],
    trait_names: typing.Optional[typing.Sequence[str]] = None,
    neutral_traits: typing.Sequence[typing.Union[int, str]] = (),
    calibrate: bool = False,
    num_bootstrap: int = 0,
    loss_rate: float = 0.0,
    min_leaf_origin_time: float = -np.inf,
    seed: typing.Optional[int] = None,
    parallel_backend: typing.Optional[str] = None,
    progress_wrap: typing.Callable = lambda x: x,
) -> pl.DataFrame:
    """Estimate per-trait gain rate and net selection from how often leaves
    carry each trait as a function of pendant branch length.

    Each leaf's pendant branch starts without the trait (unless the trait is
    shared across the parent's whole clade, in which case the leaf is
    dropped for that trait), so pendant length is an exposure window. Trait
    presence versus pendant length is fit to

        p = gain_rate * (exp(c * length) - 1) / c,  c = selection - loss_rate

    which is linear in length at neutrality and saturates under a cost.

    Only supports asexual phylogenies.

    Parameters
    ----------
    phylogeny_df : polars.DataFrame or polars.LazyFrame
        The phylogeny as a dataframe in alife standard format, with column
        `origin_time`.

        Must represent an asexual phylogeny with contiguous,
        topologically sorted ids.
    trait_masks : Sequence[str or polars.Expr]
        Column names or boolean expressions marking taxa with each trait.
        Only leaf values are used. Null values are treated as trait absent,
        with a warning.
    trait_names : Sequence[str], optional
        Labels for output rows; defaults to column names for traits given as
        str, else trait indices.
    neutral_traits : Sequence[int or str], default ()
        Indices or names of traits known to be neutral. Gain ratios are
        reported relative to their mean gain rate (leaving the focal trait
        out).
        Optional unless `calibrate` is set, which requires at least two.
        Without neutral traits, `gain_ratio` and its statistics are nan;
        gain rate and selection are still estimated.
    calibrate : bool, default False
        If True, measure pendant length by `neutral_traits` gains rather
        than by branch length, so that distortion of branch lengths shared
        with neutral traits (e.g., from tree reconstruction) cancels. See
        `alifestd_estimate_trait_gain_selection_asexual` for details.
    num_bootstrap : int, default 0
        Number of bootstrap replicates, resampling leaves with replacement
        and refitting, for standard errors, z-scores, and p-values.
    loss_rate : float, default 0.0
        Known per-unit-time trait loss rate.
    min_leaf_origin_time : float, default -inf
        Leaves originating earlier are excluded.
    seed : int, optional
        Seed for bootstrap resampling.
    parallel_backend : str, optional
        joblib backend to run bootstrap replicates in parallel, e.g., "loky"
        (processes) or "threading".
    progress_wrap : Callable, default identity
        Pass tqdm or equivalent to display bootstrap progress.

    Raises
    ------
    NotImplementedError
        If `phylogeny_df` has no "ancestor_id" column or if ids are
        non-contiguous or not topologically sorted.
    ValueError
        If `calibrate` is set with fewer than two distinct neutral traits.

    Returns
    -------
    polars.DataFrame
        One row per trait; see
        `alifestd_estimate_trait_gain_selection_asexual` for columns.

    See Also
    --------
    alifestd_estimate_trait_gain_selection_asexual :
        Pandas-based implementation.
    """
    if trait_names is None:
        trait_names = [
            mask if isinstance(mask, str) else str(i)
            for i, mask in enumerate(trait_masks)
        ]
    neutral_traits = [
        list(trait_names).index(trait) if isinstance(trait, str) else trait
        for trait in neutral_traits
    ]

    trait_masks = [
        pl.col(mask) if isinstance(mask, str) else mask for mask in trait_masks
    ]

    phylogeny_df = alifestd_try_add_ancestor_id_col_polars(phylogeny_df)
    schema_names = phylogeny_df.lazy().collect_schema().names()
    if "ancestor_id" not in schema_names:
        raise NotImplementedError(
            "alifestd_estimate_trait_gain_selection_polars only "
            "supports asexual phylogenies.",
        )

    if not phylogeny_df.lazy().limit(1).collect().is_empty():
        if not alifestd_has_contiguous_ids_polars(phylogeny_df):
            raise NotImplementedError(
                "non-contiguous ids not supported",
            )
        if not alifestd_is_topologically_sorted_polars(phylogeny_df):
            raise NotImplementedError(
                "non-topologically-sorted data not supported",
            )
        phylogeny_df = alifestd_assign_contiguous_ids_polars(
            alifestd_collapse_unifurcations_polars(
                phylogeny_df, ignore_topological_sensitivity=True
            ),
        )

    selected = (
        phylogeny_df.lazy()
        .select(
            pl.col("ancestor_id"),
            pl.col("origin_time"),
            *(
                mask.cast(pl.Boolean).alias(f"__alifestd_tgs_trait{i}")
                for i, mask in enumerate(trait_masks)
            ),
        )
        .collect()
    )

    trait_arrays = [
        _alifestd_mask_leaf_trait_inherited_polars_fill_null(
            selected[f"__alifestd_tgs_trait{i}"], name=f"trait mask {name}"
        )
        for i, name in enumerate(trait_names)
    ]

    logging.info(
        "- alifestd_estimate_trait_gain_selection_polars: "
        "fitting gain and selection...",
    )
    res = _tgs_impl(
        ancestor_ids=selected["ancestor_id"].to_numpy(),
        origin_time=selected["origin_time"].to_numpy(),
        trait_masks=trait_arrays,
        trait_names=trait_names,
        neutral_traits=neutral_traits,
        calibrate=calibrate,
        num_bootstrap=num_bootstrap,
        loss_rate=loss_rate,
        min_leaf_origin_time=min_leaf_origin_time,
        seed=seed,
        parallel_backend=parallel_backend,
        progress_wrap=progress_wrap,
    )
    return pl.from_pandas(res)


_raw_description = f"""\
{os.path.basename(__file__)} | \
(phyloframe v{get_phyloframe_version()}/joinem v{joinem.__version__})

Estimate per-trait gain rate and net selection from how often leaves carry
each trait as a function of pendant branch length.

Output is one row per trait, with columns `trait`, `num_leaves`,
`num_trait_present`, `gain_rate`, `selection`, `gain_ratio`, and
`selection_lrt_naive`; with `--num-bootstrap`, also `selection_se`,
`selection_z`, `selection_p`, `log_gain_ratio_se`, `gain_ratio_z`,
`gain_ratio_p`, and `num_bootstrap_degenerate`.

Data is assumed to be in alife standard format, with column `origin_time`.
Only supports asexual phylogenies.

Additional Notes
================
- Requires contiguous, topologically sorted ids.

- Output is a per-trait summary table, not a phylogeny.

- This CLI entrypoint is experimental and may be subject to change.

See Also
========
phyloframe.legacy._alifestd_estimate_trait_gain_selection_asexual :
    Pandas-based implementation.
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
        dfcli_module=(
            "phyloframe.legacy._alifestd_estimate_trait_gain_selection_polars"
        ),
        dfcli_version=get_phyloframe_version(),
    )
    parser.add_argument(
        "--trait-masks",
        nargs="+",
        required=True,
        type=str,
        help="Names of boolean columns marking taxa with each trait.",
    )
    parser.add_argument(
        "--neutral-traits",
        nargs="*",
        default=[],
        type=str,
        help="Names of trait columns known to be neutral (default: none).",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Measure pendant length by neutral trait gains.",
    )
    parser.add_argument(
        "--num-bootstrap",
        default=0,
        type=int,
        help="Number of bootstrap replicates (default: 0).",
    )
    parser.add_argument(
        "--bootstrap-seed",
        default=None,
        type=int,
        help="Seed for bootstrap resampling (default: None).",
    )
    parser.add_argument(
        "--loss-rate",
        default=0.0,
        type=float,
        help="Known per-unit-time trait loss rate (default: 0.0).",
    )
    parser.add_argument(
        "--min-leaf-origin-time",
        default=-np.inf,
        type=float,
        help="Exclude leaves originating earlier (default: -inf).",
    )
    parser.add_argument(
        "--parallel-backend",
        default=None,
        type=str,
        help="joblib parallel backend for bootstrap (default: None).",
    )
    return parser


if __name__ == "__main__":
    begin_prod_logging()

    parser = _create_parser()
    args, __ = parser.parse_known_args()

    try:
        with log_context_duration(
            "phyloframe.legacy._alifestd_estimate_trait_gain_selection_polars",
            logging.info,
        ):
            _run_dataframe_cli(
                base_parser=parser,
                output_dataframe_op=functools.partial(
                    alifestd_estimate_trait_gain_selection_polars,
                    trait_masks=args.trait_masks,
                    neutral_traits=args.neutral_traits,
                    calibrate=args.calibrate,
                    num_bootstrap=args.num_bootstrap,
                    seed=args.bootstrap_seed,
                    loss_rate=args.loss_rate,
                    min_leaf_origin_time=args.min_leaf_origin_time,
                    parallel_backend=args.parallel_backend,
                ),
            )
    except NotImplementedError as e:
        logging.error(
            "- polars op not yet implemented, use pandas op instead",
        )
        raise e
