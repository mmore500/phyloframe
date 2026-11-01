import typing

import pandas as pd

from .._auxlib._identity import identity
from .._auxlib._warn_experimental import warn_experimental
from ._impl import tgs_allmdc_impl, tgs_prepare_asexual, tgs_validate_args


@warn_experimental
def alifestd_tgs_allmdc_estimate_asexual(
    phylogeny_df: pd.DataFrame,
    mutate: bool = False,
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
    calibrate_size_partition_ratio: float = 2.0,
    calibrate_size_partition_min_neutral_gains: int = 20,
    num_survival_time_bins: int = 2048,
    num_bootstrap: int = 0,
    loss_rate: float = 0.0,
    selection_starts: typing.Sequence[float] = (0.0, -0.3, -1.8),
    growth_exponent_bound: float = 600.0,
    seed: typing.Optional[int] = None,
    parallel_backend: typing.Optional[str] = None,
    progress_wrap: typing.Callable = identity,
) -> pd.DataFrame:
    """Estimate per-trait gain rate and net selection from mutationally
    defined clades (MDCs) of every size.

    Unlike `alifestd_tgs_pendantmdc_estimate_asexual`, which considers only
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
      is unaffected, so disagreement between the two flags such
      distortion; and
    - nested branches share leaves, so their outcomes are not independent;
      the likelihood treats them as independent, i.e., it is a composite
      likelihood, so `selection_lrt_naive` is not calibrated as a
      chi-squared statistic, and bootstrap standard errors are likely too
      small.

    Parameters
    ----------
    phylogeny_df : pd.DataFrame
        Phylogeny in alife standard format, with column `origin_time`.
    mutate : bool, default False
        If True, allows mutation of the input dataframe.
    trait_transitions : Sequence[str]
        Names of integer columns, one per trait, giving each node's trait
        state transition on the branch from its parent: +1 for a gain, -1
        for a loss, 0 for none (and, at a root, its initial state). Make
        them from trait states at every node with
        `alifestd_mark_lineage_delta_asexual`. For leaf data, under parsimony
        without loss, a node has the trait if and only if every leaf of its
        clade does, so gains fall on the stems of mutationally defined
        clades (see `alifestd_mask_trait_mdcs_strict_asexual`). Loss events,
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
        calibration and binning; a power of two.
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
        With `calibrate`, size partitions go up the log scale only while
        each holds at least this many neutral trait gains; the first with
        fewer and all larger clades join the one below, so no size partition
        is left without neutral gains to calibrate against.
    num_survival_time_bins : int, default 2048
        Number of equal-occupancy survival-time bins. In the fit, each
        branch's survival time is replaced by the mean survival time of its
        bin. As the survival term is convex in survival time, this
        understates its average, so too few bins bias gain rate upward.
    num_bootstrap : int, default 0
        Number of bootstrap replicates, resampling branches with
        replacement and refitting, for standard errors,
        z-scores, and p-values. Calibration neighborhood sizes are held
        fixed across replicates. Around 200 replicates suffice for standard
        errors (Efron and Tibshirani, 1993, *An Introduction to the
        Bootstrap*, Chapter 6); a few dozen give rough ones.
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
    pd.DataFrame
        One row per trait, with columns `trait`, `num_leaves` and
        `num_trait_present` (leaf branches that started without the trait,
        and those that gained it), `num_clades` and `num_clade_gains` (the
        same for inner branches), `gain_rate`, `selection`, `gain_ratio`
        (gain rate relative to the mean of neutral traits', leaving the
        focal trait out), and `selection_lrt_naive`. With bootstrap
        replicates, also the pendant estimator's bootstrap columns.

    Raises
    ------
    ValueError
        If arguments are invalid, e.g., `calibrate` with fewer than two
        neutral traits, or a `trait_transitions` column is missing or gives
        states other than 0 and 1.
    NotImplementedError
        If a `trait_transitions` column is not integer, or has loss events,
        i.e., -1 values.

    Notes
    -----
    Unifurcations are bypassed. Branches with zero or undefined length
    (including roots) are excluded. Bootstrap replicates resample branches
    independently, ignoring correlation through shared ancestry.

    Input dataframe is not mutated by this operation unless `mutate` set
    True. If mutate set True, operation does not occur in place; still use
    return value to get transformed phylogeny dataframe.

    See Also
    --------
    alifestd_tgs_allmdc_estimate_polars :
        Polars-based implementation.
    alifestd_tgs_pendantmdc_estimate_asexual :
        Leaf pendant branches only.
    """
    trait_names, neutral_traits = tgs_validate_args(
        trait_transitions=trait_transitions,
        neutral_traits=neutral_traits,
        calibrate=calibrate,
        calibrate_neighborhood_fractions=calibrate_neighborhood_fractions,
        calibrate_num_partitions=calibrate_num_partitions,
    )
    branches = tgs_prepare_asexual(
        phylogeny_df,
        mutate,
        trait_transitions=trait_transitions,
    )
    return tgs_allmdc_impl(
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
        growth_exponent_bound=growth_exponent_bound,
        seed=seed,
        parallel_backend=parallel_backend,
        progress_wrap=progress_wrap,
    )
