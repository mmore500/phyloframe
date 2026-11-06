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
from ._impl import (
    tgs_allmdc_count_episodes,
    tgs_build_exposure_df_polars,
    tgs_impl,
    tgs_validate_args,
)


@warn_experimental
def alifestd_tgs_allmdc_estimate_polars(
    phylogeny_df: pl.DataFrame,
    *,
    trait_transitions: typing.Sequence[str],
    neutral_traits: typing.Sequence[str],
    calibration_partitions: typing.Optional[str] = None,
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
    calibrate_max_rate_ratio: float = np.inf,
    length_rtol: float = 1e-3,
    num_bootstrap: int = 0,
    loss_rate: float = 0.0,
    selection_coef_starts: typing.Sequence[float] = (0.0, -0.3, -1.8),
    growth_exponent_bound: float = 600.0,
    seed: typing.Optional[int] = None,
    parallel_backend: typing.Optional[str] = None,
    progress_wrap: typing.Callable = identity,
) -> pl.DataFrame:
    """Estimate per-trait gain rate and selection coefficient from mutationally
    defined clades (MDCs) of every size.

    Unlike `alifestd_tgs_pendantmdc_estimate_polars`, which considers only
    leaf pendant branches (i.e., single-leaf MDCs), considers inner
    branches too, so that trait gains shared by a multi-leaf clade also
    inform the fit.

    Here, an MDC is a mutationally defined clade: a maximal clade whose
    leaves all carry the trait, which, under parsimony without loss, gained
    it on its stem branch. Each branch whose parent lacks the trait is an
    exposure episode: it started without the trait, so its length is an
    exposure window, and it gained the trait if its end node carries it.
    Every exposure episode, pendant or inner, is fit to

        p = gain_rate * (exp(c * length) - 1) / c * exp(c * survival),
        c = selection_coef - loss_rate

    where `survival` is the episode's survival duration, measured on the
    tree from its end node to the latest sampled leaf below it (zero for a
    leaf): after a gain, its carrier lineage must persist that much longer
    to be seen at every leaf of the clade, which a cost makes less likely.
    Equivalently, writing `eff(t) = (exp(c * t) - 1) / c` for the effective
    exposure of a window of length `t`, `p = gain_rate * (eff(length +
    survival) - eff(survival))`: the exposure of the window from the
    branch's start to the clade's latest leaf, minus that of the window
    from the branch's end.

    Only supports asexual phylogenies.

    Parameters
    ----------
    phylogeny_df : polars.DataFrame or polars.LazyFrame
        The phylogeny as a dataframe in alife standard format; see Input
        Schema.
    trait_transitions : Sequence[str]
        Names of integer columns, one per trait, giving each node's trait
        state transition on the branch from its parent (see Input Schema).
        Output rows are labeled by these column names.
    neutral_traits : Sequence[str]
        Names, among `trait_transitions`, of traits known to be neutral. At
        least two are required: they calibrate branch lengths against one
        another, and their mean gain rate, leaving the focal trait out, is
        the reference for gain ratios.
    size_partitions : str, optional
        Name of an integer column giving each node's clade size partition
        (see Input Schema), e.g., from
        `alifestd_tgs_mark_clade_size_partitions_polars`. Branch lengths are
        calibrated within each size partition, as neutral gains per unit
        length may differ with clade size (e.g., between leaves and inner
        nodes). If not given, all nodes form one size partition, i.e., no
        calibration across clade sizes.
    calibrate_neighborhood_fractions : Sequence[float], default (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0)
        Candidate calibration neighborhood sizes, as fractions of each size
        partition's exposure episodes, similar in branch length and origin
        time; see `alifestd_tgs_pendantmdc_estimate_polars`. Pass (1.0,),
        one neighborhood of each size partition's episodes, to disable
        calibration within size partitions.
    calibrate_num_partitions : int, default 4096
        Maximum number of equal-occupancy partitions of each size
        partition's exposure episodes, by ranked branch length and origin
        time, for calibration and fitting; a power of two.
    calibrate_max_rate_ratio : float, default np.inf
        Cap on the neutral gain rate ratios that rescale branch lengths in
        calibration; see `alifestd_tgs_pendantmdc_estimate_polars`.
    num_bootstrap : int, default 0
        Number of bootstrap replicates, resampling exposure episodes with
        replacement and refitting, for standard errors, z-scores, and
        p-values. Calibration neighborhood size is held constant. More than
        about 200 replicates are rarely needed for standard errors; see
        Efron and Tibshirani (1993), An Introduction to the Bootstrap
        (Chapman & Hall).
    loss_rate : float, default 0.0
        Known per-unit-time trait loss rate.
    selection_coef_starts : Sequence[float], default (0.0, -0.3, -1.8)
        Starting values for the selection coefficient, in units of 1 / mean
        branch length; the best of the resulting fits is kept. See `tgs_fit`.
    growth_exponent_bound : float, default 600.0
        Bound on carrier growth exponents, such as the selection coefficient
        minus loss rate, times branch length, keeping optimizer trial steps
        finite. See `tgs_fit`.
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
        One row per trait; see Output Schema.

    Raises
    ------
    ValueError
        If there are no exposure episodes (i.e., branches to fit), if an
        argument is invalid, or if a `trait_transitions` or
        `size_partitions` column is missing, or a `trait_transitions` column
        gives states other than 0 and 1.
    NotImplementedError
        If fewer than two neutral traits are given; if `phylogeny_df` is not
        asexual, or its ids are non-contiguous or not topologically sorted;
        if a `trait_transitions` column is not integer, has null values, or
        has loss events, i.e., -1 values; or if the `size_partitions` column
        is not integer or has null values.

    Input Schema
    ------------
    'id' : integer
        Taxon identifier; contiguous and topologically sorted.
    'ancestor_id' : integer
        Parent's taxon identifier (or own, for a root).
    'origin_time' : float
        Time of each taxon's origin, from which branch lengths and survival
        durations are taken.
    `trait_transitions` columns : integer
        Each node's trait state transition on the branch from its parent; see
        `alifestd_tgs_pendantmdc_estimate_polars`.
    `size_partitions` column : integer, optional
        Each node's clade size partition, for calibration.

    Output Schema
    -------------
    As for `alifestd_tgs_pendantmdc_estimate_polars`, with also:

    'num_clades' : integer
        Inner exposure episodes, i.e., inner branches that started without
        the trait.
    'num_clade_gains' : integer
        Of those, the number that gained the trait.

    Here, 'num_leaves' and 'num_trait_present' count leaf exposure
    episodes, and 'gain_rate' is per unit branch length.

    Notes
    -----
    Unifurcations are collapsed (see `alifestd_collapse_unifurcations_polars`).

    Survival durations take the tree's branch lengths at face value, so
    distortion of inner branch lengths (e.g., from reconstruction) also
    distorts the survival term, which neutral calibration does not correct.
    The pendant estimator, which uses no inner branch lengths, is
    unaffected, so disagreement between the two may suggest such
    distortion.

    Nested exposure episodes share leaves, so their outcomes are not
    independent, but the likelihood treats them as independent (a composite
    likelihood). So, `selection_coef_lrt_naive` and `gain_ratio_lrt_naive`
    are not calibrated as chi-squared statistics, and every bootstrap
    statistic (standard errors, z-scores, and p-values), as replicates
    resample episodes independently, is likely overconfident. No bootstrap
    that accounts for nesting (e.g., resampling whole clades) is yet
    provided.

    See Also
    --------
    alifestd_tgs_mark_clade_size_partitions_polars :
        Marks clade size partitions, for `size_partitions`.
    alifestd_tgs_pendantmdc_estimate_polars :
        Leaf pendant branches only.
    """
    trait_names, neutral_traits = tgs_validate_args(
        trait_transitions=trait_transitions,
        neutral_traits=neutral_traits,
        calibrate_neighborhood_fractions=calibrate_neighborhood_fractions,
        calibrate_max_rate_ratio=calibrate_max_rate_ratio,
        length_rtol=length_rtol,
    )
    exposure_df = tgs_build_exposure_df_polars(
        phylogeny_df,
        trait_transitions=trait_transitions,
        caller="alifestd_tgs_allmdc_estimate_polars",
        calibration_partitions=calibration_partitions,
    )
    res = tgs_impl(
        exposure_df,
        counts_df=tgs_allmdc_count_episodes(
            trait_names=trait_names, exposure_df=exposure_df
        ),
        trait_names=trait_names,
        neutral_traits=neutral_traits,
        calibrate_neighborhood_fractions=calibrate_neighborhood_fractions,
        calibrate_max_rate_ratio=calibrate_max_rate_ratio,
        length_rtol=length_rtol,
        num_bootstrap=num_bootstrap,
        loss_rate=loss_rate,
        selection_coef_starts=selection_coef_starts,
        growth_exponent_bound=growth_exponent_bound,
        seed=seed,
        parallel_backend=parallel_backend,
        progress_wrap=progress_wrap,
    )
    return pl.from_pandas(res)


_raw_description = f"""\
{os.path.basename(__file__)} | \
(phyloframe v{get_phyloframe_version()}/joinem v{joinem.__version__})

Estimate per-trait gain rate and selection coefficient from how often each
exposure episode (branch starting without a trait) gains it, as a function
of branch length and survival duration (from the branch's end to its latest
sampled leaf), using pendant and inner branches.

Data is assumed to be in alife standard format.
Only supports asexual phylogenies.


Input Schema: Required Columns
==============================
'id' : integer
    Taxon identifier; contiguous and topologically sorted.

'ancestor_id' : integer
    Parent's taxon identifier (or own, for a root).

'origin_time' : float
    Time of each taxon's origin.

`--trait-transitions` columns : integer
    Each node's trait state transition on the branch from its parent: +1
    for a gain, 0 for none, and, at a root, its initial state.


Input Schema: Optional Columns
==============================
`--size-partitions` column : integer
    Each node's clade size partition, for calibration (see
    phyloframe.legacy._alifestd_tgs_mark_clade_size_partitions_polars).


Output Schema
=============
As for phyloframe.legacy._alifestd_tgs_pendantmdc_estimate_polars, with also
'num_clades' and 'num_clade_gains' : integer, the inner exposure episodes and
how many gained the trait.


Additional Notes
================
- Requires contiguous, topologically sorted ids.

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
        "--size-partitions",
        default=None,
        type=str,
        help=(
            "Name of integer column of clade size partitions, within which "
            "to calibrate (default: none, i.e., one partition)."
        ),
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
        "--selection-coef-starts",
        default=(0.0, -0.3, -1.8),
        nargs="+",
        type=float,
        help=(
            "Starting values for the selection coefficient, in units of "
            "1 / mean branch length (default: 0.0 -0.3 -1.8)."
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
                    calibrate_neighborhood_fractions=(
                        args.calibrate_neighborhood_fractions
                    ),
                    calibrate_num_partitions=args.calibrate_num_partitions,
                    calibrate_max_rate_ratio=args.calibrate_max_rate_ratio,
                    size_partitions=args.size_partitions,
                    num_bootstrap=args.num_bootstrap,
                    seed=args.bootstrap_seed,
                    loss_rate=args.loss_rate,
                    selection_coef_starts=args.selection_coef_starts,
                    growth_exponent_bound=args.growth_exponent_bound,
                    parallel_backend=args.parallel_backend,
                ),
            )
    except NotImplementedError as e:
        logging.error(
            "- polars op not yet implemented, use pandas op instead",
        )
        raise e
