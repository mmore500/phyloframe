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
from .._auxlib._identity import identity
from .._auxlib._log_context_duration import log_context_duration
from .._auxlib._warn_experimental import warn_experimental
from ._impl import tgs_allmdc_impl, tgs_prepare_polars, tgs_validate_args


@warn_experimental
def alifestd_tgs_allmdc_estimate_polars(
    phylogeny_df: pl.DataFrame,
    *,
    trait_transitions: typing.Sequence[str],
    trait_names: typing.Optional[typing.Sequence[str]] = None,
    neutral_traits: typing.Sequence[typing.Union[int, str]] = (),
    calibrate: bool = False,
    calibrate_neighborhood_fractions: typing.Sequence[float] = (
        0.005,
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
    ),
    calibrate_num_partitions: int = 4096,
    calibrate_size_partition_ratio: float = 2.0,
    calibrate_size_partition_min_neutral_gains: int = 20,
    num_survival_time_bins: int = 2048,
    num_bootstrap: int = 0,
    loss_rate: float = 0.0,
    selection_starts: typing.Sequence[float] = (0.0, -0.3, -1.8),
    max_growth_exponent: float = 600.0,
    seed: typing.Optional[int] = None,
    parallel_backend: typing.Optional[str] = None,
    progress_wrap: typing.Callable = identity,
) -> pl.DataFrame:
    """Estimate per-trait gain rate and net selection from every branch,
    pendant and inner, i.e., from mutationally defined clades (MDCs) of
    every size, from how often each branch's clade carries the trait as a
    function of branch length and survival time.

    Each branch's gain probability is discounted for the time, measured on
    the tree from the branch's end to its latest sampled leaf, that a
    carrier lineage must persist to be seen at every leaf of its clade. See
    `alifestd_tgs_allmdc_estimate_asexual` for the model.

    Only supports asexual phylogenies.

    Parameters
    ----------
    phylogeny_df : polars.DataFrame or polars.LazyFrame
        The phylogeny as a dataframe in alife standard format, with column
        `origin_time`.

        Must represent an asexual phylogeny with contiguous,
        topologically sorted ids.
    trait_transitions : Sequence[str]
        Names of integer columns, one per trait, giving each node's trait
        state transition on the branch from its parent: +1 for a gain, -1
        for a loss, 0 for none (and, at a root, its initial state). Make
        them from trait states at every node with
        `alifestd_mark_lineage_delta_polars`. For leaf data, under parsimony
        without loss, a node has the trait if and only if every leaf of its
        clade does, so gains fall on the stems of mutationally defined
        clades (see `alifestd_mask_trait_mdcs_polars`). Losses are not yet
        supported.
    trait_names : Sequence[str], optional
        Labels for output rows; defaults to `trait_transitions`.
    neutral_traits : Sequence[int or str], default ()
        Indices or names of traits known to be neutral. Gain ratios are
        reported relative to their mean gain rate (leaving the focal trait
        out). Optional unless `calibrate` is set, which requires at least
        two.
    calibrate : bool, default False
        If True, calibrate branch lengths against `neutral_traits` gains
        within neighborhoods of similar branches.
    calibrate_neighborhood_fractions : Sequence[float], default (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0)
        Candidate calibration neighborhood sizes, as fractions of branches.
    calibrate_num_partitions : int, default 4096
        Maximum number of partitions of each size partition's branches for
        calibration and binning; a power of two.
    calibrate_size_partition_ratio : float, default 2.0
        Ratio between successive size partitions, which group branches by
        clade size for calibration, e.g., 2 for size partitions of 1, 2,
        3-4, 5-8, ... leaves; see
        `alifestd_tgs_allmdc_estimate_asexual`.
    calibrate_size_partition_min_neutral_gains : int, default 20
        With `calibrate`, larger clades join the size partition below once
        one would hold fewer than this many neutral trait gains; see
        `alifestd_tgs_allmdc_estimate_asexual`.
    num_survival_time_bins : int, default 2048
        Number of equal-occupancy survival-time bins, over which each
        partition's survival term is averaged.
    num_bootstrap : int, default 0
        Number of bootstrap replicates, resampling branches with
        replacement and refitting.
    loss_rate : float, default 0.0
        Known per-unit-time trait loss rate.
    selection_starts : Sequence[float], default (0.0, -0.3, -1.8)
        Starting values for selection, in units of 1 / mean branch length.
    max_growth_exponent : float, default 600.0
        Cap on carrier growth exponents.
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
        If `phylogeny_df` has no "ancestor_id" column, if ids are
        non-contiguous or not topologically sorted, or if a
        `trait_transitions` column is not integer, has null values, or has
        losses (-1).
    ValueError
        If `calibrate` is set with fewer than two distinct neutral traits,
        if a `trait_transitions` column is missing or has values other than
        +1, -1, or 0, or gives states other than 0 and 1, if `trait_names`
        and `trait_transitions` differ in length, if
        `num_survival_time_bins` is less than 1, if
        `calibrate_size_partition_ratio` is not greater than 1, if
        `calibrate_size_partition_min_neutral_gains` is less than 1, if
        `calibrate_neighborhood_fractions` is empty or has a value outside
        (0, 1], or if `calibrate_num_partitions` is not a power of two.

    Returns
    -------
    polars.DataFrame
        One row per trait; see
        `alifestd_tgs_allmdc_estimate_asexual` for columns.

    See Also
    --------
    alifestd_tgs_allmdc_estimate_asexual :
        Pandas-based implementation.
    alifestd_tgs_pendantmdc_estimate_polars :
        Leaf pendant branches only.
    """
    trait_names, neutral_traits = tgs_validate_args(
        trait_transitions=trait_transitions,
        trait_names=trait_names,
        neutral_traits=neutral_traits,
        calibrate=calibrate,
        calibrate_neighborhood_fractions=calibrate_neighborhood_fractions,
        calibrate_num_partitions=calibrate_num_partitions,
    )
    branches = tgs_prepare_polars(
        phylogeny_df,
        trait_transitions=trait_transitions,
        trait_names=trait_names,
        caller="alifestd_tgs_allmdc_estimate_polars",
    )

    logging.info(
        "- alifestd_tgs_allmdc_estimate_polars: "
        "fitting gain and selection...",
    )
    res = tgs_allmdc_impl(
        branches,
        trait_names=trait_names,
        neutral_traits=neutral_traits,
        calibrate=calibrate,
        calibrate_neighborhood_fractions=calibrate_neighborhood_fractions,
        calibrate_num_partitions=calibrate_num_partitions,
        calibrate_size_partition_ratio=calibrate_size_partition_ratio,
        calibrate_size_partition_min_neutral_gains=calibrate_size_partition_min_neutral_gains,
        num_survival_time_bins=num_survival_time_bins,
        num_bootstrap=num_bootstrap,
        loss_rate=loss_rate,
        selection_starts=selection_starts,
        max_growth_exponent=max_growth_exponent,
        seed=seed,
        parallel_backend=parallel_backend,
        progress_wrap=progress_wrap,
    )
    return pl.from_pandas(res)


_raw_description = f"""\
{os.path.basename(__file__)} | \
(phyloframe v{get_phyloframe_version()}/joinem v{joinem.__version__})

Estimate per-trait gain rate and net selection from how often each branch's
clade carries each trait as a function of branch length and survival time
(from the branch's end to its latest sampled leaf), using pendant and inner
branches.

Output is one row per trait, with columns `trait`, `num_leaves`,
`num_trait_present`, `num_clades`, `num_clade_gains`, `gain_rate`,
`selection`, `gain_ratio`, and `selection_lrt_naive`; with `--num-bootstrap`,
also `selection_se`, `selection_z`, `selection_p`, `log_gain_ratio_se`,
`gain_ratio_z`, `gain_ratio_p`, and `num_bootstrap_degenerate`.

Data is assumed to be in alife standard format, with column `origin_time`.
Only supports asexual phylogenies.

Additional Notes
================
- Requires contiguous, topologically sorted ids.

- Output is a per-trait summary table, not a phylogeny.

- This CLI entrypoint is experimental and may be subject to change.

See Also
========
phyloframe.legacy._alifestd_tgs_allmdc_estimate_asexual :
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
            "phyloframe.legacy._alifestd_tgs_allmdc_estimate_polars"
        ),
        dfcli_version=get_phyloframe_version(),
    )
    parser.add_argument(
        "--trait-transitions",
        nargs="+",
        required=True,
        type=str,
        help=(
            "Names of integer columns giving each trait's state transition "
            "on the branch into each node (+1 gain, -1 loss, 0 none)."
        ),
    )
    parser.add_argument(
        "--neutral-traits",
        nargs="*",
        default=[],
        type=str,
        help=(
            "Names of trait transition columns of traits known to be neutral "
            "(default: none)."
        ),
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Measure branch length by neutral trait gains.",
    )
    parser.add_argument(
        "--calibrate-neighborhood-fractions",
        default=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
        nargs="+",
        type=float,
        help=(
            "Candidate calibration neighborhood sizes, as fractions of "
            "branches (default: 0.005 0.01 0.025 0.05 0.1 0.25 0.5 1.0)."
        ),
    )
    parser.add_argument(
        "--calibrate-num-partitions",
        default=4096,
        type=int,
        help=(
            "Maximum number of branch partitions per size partition, a "
            "power of two (default: 4096)."
        ),
    )
    parser.add_argument(
        "--calibrate-size-partition-ratio",
        default=2.0,
        type=float,
        help=(
            "Ratio between successive size partitions, which group branches "
            "by clade size for calibration, e.g., 2 for 1, 2, 3-4, 5-8, ... "
            "leaves (default: 2.0)."
        ),
    )
    parser.add_argument(
        "--calibrate-size-partition-min-neutral-gains",
        default=20,
        type=int,
        help=(
            "Minimum neutral trait gains per size partition; larger clades "
            "join the size partition below (default: 20)."
        ),
    )
    parser.add_argument(
        "--num-survival-time-bins",
        default=2048,
        type=int,
        help="Number of survival-time bins (default: 2048).",
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
        "--selection-starts",
        default=(0.0, -0.3, -1.8),
        nargs="+",
        type=float,
        help=(
            "Starting values for selection, in units of 1 / mean branch "
            "length (default: 0.0 -0.3 -1.8)."
        ),
    )
    parser.add_argument(
        "--max-growth-exponent",
        default=600.0,
        type=float,
        help="Cap on carrier growth exponents (default: 600.0).",
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
            "phyloframe.legacy._alifestd_tgs_allmdc_estimate_polars",
            logging.info,
        ):
            _run_dataframe_cli(
                base_parser=parser,
                output_dataframe_op=functools.partial(
                    alifestd_tgs_allmdc_estimate_polars,
                    trait_transitions=args.trait_transitions,
                    neutral_traits=args.neutral_traits,
                    calibrate=args.calibrate,
                    calibrate_neighborhood_fractions=(
                        args.calibrate_neighborhood_fractions
                    ),
                    calibrate_num_partitions=args.calibrate_num_partitions,
                    calibrate_size_partition_ratio=(
                        args.calibrate_size_partition_ratio
                    ),
                    calibrate_size_partition_min_neutral_gains=(
                        args.calibrate_size_partition_min_neutral_gains
                    ),
                    num_survival_time_bins=args.num_survival_time_bins,
                    num_bootstrap=args.num_bootstrap,
                    seed=args.bootstrap_seed,
                    loss_rate=args.loss_rate,
                    selection_starts=args.selection_starts,
                    max_growth_exponent=args.max_growth_exponent,
                    parallel_backend=args.parallel_backend,
                ),
            )
    except NotImplementedError as e:
        logging.error(
            "- polars op not yet implemented, use pandas op instead",
        )
        raise e
