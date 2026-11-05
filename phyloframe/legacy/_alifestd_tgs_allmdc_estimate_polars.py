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
from .._auxlib._identity import identity
from .._auxlib._log_context_duration import log_context_duration
from .._auxlib._warn_experimental import warn_experimental
from ._impl import tgs_allmdc_impl, tgs_prepare_polars, tgs_validate_args


@warn_experimental
def alifestd_tgs_allmdc_estimate_polars(
    phylogeny_df: pl.DataFrame,
    *,
    trait_transitions: typing.Sequence[str],
    neutral_traits: typing.Sequence[str] = (),
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
    calibrate_max_rate_ratio: float = np.inf,
    calibrate_size_partition_ratio: float = 2.0,
    calibrate_size_partition_min_neutral_gains: int = 20,
    num_bootstrap: int = 0,
    loss_rate: float = 0.0,
    selection_starts: typing.Sequence[float] = (0.0, -0.3, -1.8),
    growth_exponent_bound: float = 600.0,
    seed: typing.Optional[int] = None,
    parallel_backend: typing.Optional[str] = None,
    progress_wrap: typing.Callable = identity,
) -> pl.DataFrame:
    """Estimate per-trait gain rate and net selection from mutationally
    defined clades (MDCs) of every size.

    Unlike `alifestd_tgs_pendantmdc_estimate_polars`, which considers only
    leaf pendant branches (i.e., single-leaf MDCs), considers inner
    branches too, so that trait gains shared by a multi-leaf clade also
    inform the fit.

    Here, an MDC is a mutationally defined clade: a maximal clade whose
    leaves all carry the trait, which, under parsimony without loss, gained
    it on its stem branch. Each branch whose parent lacks the trait started
    without it, so its length is an exposure window; the branch gained the
    trait if its end node carries it. Every such branch, pendant or inner,
    is fit to

        p = gain_rate * (exp(c * length) - 1) / c * exp(c * survival),
        c = selection - loss_rate

    where `survival` is the branch's survival time, measured on the tree
    from its end node to the latest sampled leaf below it (zero for a
    leaf): after a gain on the branch, its carrier lineage must persist
    that much longer to be seen at every leaf of the clade, which a cost
    makes less likely. Equivalently, writing `eff(t) = (exp(c * t) - 1) /
    c` for the effective exposure of a window of length `t`, `p =
    gain_rate * (eff(length + survival) - eff(survival))`: the exposure of
    the window from the branch's start to the clade's latest leaf, minus
    that of the window from the branch's end.

    Notes on interpretation:

    - survival time takes the tree's branch lengths at face value, so
      distortion of inner branch lengths (e.g., from reconstruction) also
      distorts the survival term, which neutral calibration does not
      correct; the pendant estimator, which uses no inner branch lengths,
      is unaffected, so disagreement between the two may suggest such
      distortion; and
    - nested branches share leaves, so their outcomes are not independent, but
      the likelihood treats them as independent (a composite likelihood). So,
      `selection_lrt_naive` and `gain_ratio_lrt_naive` are not calibrated as
      chi-squared statistics, and every bootstrap statistic (standard errors,
      z-scores, and p-values), as replicates resample branches independently,
      is likely overconfident. No bootstrap that accounts for nesting (e.g.,
      resampling whole clades) is yet provided.

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
        `alifestd_mark_lineage_delta_polars`, with `basal_value=False`
        to give each root's initial state. For leaf data, under parsimony
        without loss, a node has the trait if and only if every leaf of its
        clade does, so gains fall on the stems of mutationally defined
        clades (see `alifestd_mask_trait_mdcs_strict_polars`). Loss events,
        i.e., -1 values, are not yet supported. Output rows are labeled by
        these column names.
    neutral_traits : Sequence[str], default ()
        Names, among `trait_transitions`, of traits known to be neutral.
        Used to calibrate branch lengths (with `calibrate`), which requires
        at least two, and as the reference for gain ratios.
    calibrate : bool, default False
        If True, rescale branch lengths by neutral trait gains, as in the
        pendant estimator, within neighborhoods of branches of the same
        size partition (see `calibrate_size_partition_ratio`) similar in
        length and origin time. So, distortion of branch
        lengths shared with neutral traits, including any difference
        between inner and pendant branches, cancels.
    calibrate_neighborhood_fractions : Sequence[float], default (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0)
        Candidate calibration neighborhood sizes, as fractions of each size
        partition's branches; see the pendant estimator.
    calibrate_num_partitions : int, default 4096
        Maximum number of equal-occupancy partitions of each size
        partition's branches, by ranked length and origin time, for
        calibration and fitting; a power of two.
    calibrate_max_rate_ratio : float, default np.inf
        Cap on the neutral gain rate ratios that rescale branch lengths in
        calibration; see the pendant estimator.
    calibrate_size_partition_ratio : float, default 2.0
        Ratio between successive size partitions, which group branches by
        clade size for calibration: a branch subtending `n` leaves falls in
        size partition ceil(log(n) / log(ratio)), so the default gives size
        partitions of 1, 2, 3-4, 5-8, 9-16, ... leaves. Branches are
        partitioned and calibrated within their size partition, as neutral
        gains per unit length may differ with clade size; the model itself
        is the same for every size partition. Larger ratios give fewer,
        larger size partitions.
    calibrate_size_partition_min_neutral_gains : int, default 20
        With `calibrate`, inner size partitions go up the log scale only
        while each holds at least this many neutral trait gains; the first
        with fewer and all larger clades join the one below (or, if the
        first inner size partition has too few, all inner branches form
        one), so no inner size partition is left without neutral gains to
        calibrate against. Leaves always keep their own size partition, and
        clade sizes absent from the tree leave no empty size partitions.
    num_bootstrap : int, default 0
        Number of bootstrap replicates, resampling branches with
        replacement and refitting, for standard errors,
        z-scores, and p-values. Calibration neighborhood sizes are held
        fixed across replicates. More than about 200 replicates are
        rarely needed for standard errors; see Efron and Tibshirani (1993), An
        Introduction to the Bootstrap (Chapman & Hall).
    loss_rate : float, default 0.0
        Known per-unit-time trait loss rate.
    selection_starts : Sequence[float], default (0.0, -0.3, -1.8)
        Starting values for selection, in units of 1 / mean branch length;
        the best of the resulting fits is kept. See `tgs_fit`.
    growth_exponent_bound : float, default 600.0
        Bound on carrier growth exponents, such as (selection - loss rate) *
        branch length, keeping optimizer trial steps finite. See `tgs_fit`.
    seed : int, default None
        Seed for bootstrap resampling; if None, drawn from numpy's global
        random state.
    parallel_backend : str, default None
        joblib backend to run bootstrap replicates in parallel, e.g., "loky"
        (processes) or "threading"; if None, runs them sequentially.
    progress_wrap : Callable, default identity
        Pass tqdm or equivalent to display bootstrap progress.

    Returns
    -------
    polars.DataFrame
        One row per trait, with columns `trait`, `num_leaves` and
        `num_trait_present` (leaf branches that started without the trait,
        and those that gained it), `num_clades` and `num_clade_gains` (the
        same for inner branches), `gain_rate`, `selection`, `gain_ratio`
        (gain rate relative to the mean of neutral traits', leaving the
        focal trait out), `selection_lrt_naive`, and `gain_ratio_lrt_naive`
        (likelihood-ratio statistics for selection = 0 and for gain ratio =
        1; see the pendant estimator). With bootstrap replicates, also the
        pendant estimator's bootstrap columns.

    Raises
    ------
    ValueError
        If arguments are invalid, e.g., `calibrate` with fewer than two
        neutral traits, or a `trait_transitions` column is missing or gives
        states other than 0 and 1, or if a non-root node has zero,
        negative, or undefined (nan) branch length.
    NotImplementedError
        If `phylogeny_df` has no "ancestor_id" column, if ids are
        non-contiguous or not topologically sorted, or if a
        `trait_transitions` column is not integer, has null values, or has
        loss events, i.e., -1 values.

    Notes
    -----
    Unifurcations are collapsed (see `alifestd_collapse_unifurcations_polars`).
    Bootstrap replicates resample branches independently, ignoring
    correlation through shared ancestry.

    See Also
    --------
    alifestd_tgs_pendantmdc_estimate_polars :
        Leaf pendant branches only.
    """
    trait_names, neutral_traits = tgs_validate_args(
        trait_transitions=trait_transitions,
        neutral_traits=neutral_traits,
        calibrate=calibrate,
        calibrate_neighborhood_fractions=calibrate_neighborhood_fractions,
        calibrate_num_partitions=calibrate_num_partitions,
        calibrate_max_rate_ratio=calibrate_max_rate_ratio,
    )
    exposure_df = tgs_prepare_polars(
        phylogeny_df,
        trait_transitions=trait_transitions,
        caller="alifestd_tgs_allmdc_estimate_polars",
    )

    logging.info(
        "- alifestd_tgs_allmdc_estimate_polars: "
        "fitting gain and selection...",
    )
    res = tgs_allmdc_impl(
        exposure_df,
        trait_names=trait_names,
        neutral_traits=neutral_traits,
        calibrate=calibrate,
        calibrate_neighborhood_fractions=calibrate_neighborhood_fractions,
        calibrate_num_partitions=calibrate_num_partitions,
        calibrate_max_rate_ratio=calibrate_max_rate_ratio,
        calibrate_size_partition_ratio=calibrate_size_partition_ratio,
        calibrate_size_partition_min_neutral_gains=(
            calibrate_size_partition_min_neutral_gains
        ),
        num_bootstrap=num_bootstrap,
        loss_rate=loss_rate,
        selection_starts=selection_starts,
        growth_exponent_bound=growth_exponent_bound,
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
`selection`, `gain_ratio`, `selection_lrt_naive`, and `gain_ratio_lrt_naive`;
with `--num-bootstrap`, also `selection_se`, `selection_z`, `selection_p`,
`log_gain_ratio_se`, `gain_ratio_z`, `gain_ratio_p`, and
`num_bootstrap_degenerate`.

Data is assumed to be in alife standard format, with column `origin_time`.
Only supports asexual phylogenies.

Additional Notes
================
- Requires contiguous, topologically sorted ids.

- Output is a per-trait summary table, not a phylogeny.

- This CLI entrypoint is experimental and may be subject to change.

See Also
========
phyloframe.legacy._alifestd_tgs_pendantmdc_estimate_polars :
    Leaf pendant branches only.
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
        "--calibrate-max-rate-ratio",
        default=np.inf,
        type=float,
        help=(
            "Cap on calibration's neutral gain rate ratios "
            "(default: uncapped)."
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
        "--growth-exponent-bound",
        default=600.0,
        type=float,
        help="Bound on carrier growth exponents (default: 600.0).",
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
                    calibrate_max_rate_ratio=args.calibrate_max_rate_ratio,
                    calibrate_size_partition_ratio=(
                        args.calibrate_size_partition_ratio
                    ),
                    calibrate_size_partition_min_neutral_gains=(
                        args.calibrate_size_partition_min_neutral_gains
                    ),
                    num_bootstrap=args.num_bootstrap,
                    seed=args.bootstrap_seed,
                    loss_rate=args.loss_rate,
                    selection_starts=args.selection_starts,
                    growth_exponent_bound=args.growth_exponent_bound,
                    parallel_backend=args.parallel_backend,
                ),
            )
    except NotImplementedError as e:
        logging.error(
            "- polars op not yet implemented, use pandas op instead",
        )
        raise e
