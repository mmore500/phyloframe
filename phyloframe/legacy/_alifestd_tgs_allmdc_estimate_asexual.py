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
) -> pd.DataFrame:
    """Estimate per-trait gain rate and net selection from every branch,
    pendant and inner, from how often each branch's clade carries the trait
    as a function of branch length and survival time.

    Here, an MDC is a mutationally defined clade: a maximal clade whose
    leaves all carry the trait, which, under parsimony without loss, gained
    it on its stem branch. This estimator uses MDCs of every size, single
    leaves included.

    Extends `alifestd_tgs_pendantmdc_estimate_asexual`, which
    uses leaf pendant branches only, to inner branches, so that trait gains
    shared by a multi-leaf clade also inform the fit. Each branch whose
    parent's clade does not carry the trait at every leaf started without
    it, so branch length is an exposure window; the branch gained the trait
    (under parsimony, without loss) if its own clade carries it at every
    leaf. Every branch, pendant or inner, is fit to

        p = gain_rate * (exp(c * length) - 1) / c * exp(c * survival),
        c = selection - loss_rate

    where `survival` is the branch's survival time, measured on the tree
    from its end node to the latest sampled leaf below it (zero for a
    leaf): after a gain on the branch, its carrier lineage must persist
    that much longer to be seen at every leaf of the clade, which a cost
    makes less likely. Equivalently, with effective length `eff(t) =
    (exp(c * t) - 1) / c`, `p = gain_rate * (eff(length + survival) -
    eff(survival))`, the exposure the branch adds to its clade's.

    Notes on interpretation:

    - leaves are scored as in the pendant estimator, so with only leaf
      branches (e.g., a star tree), estimates match it;
    - survival time takes the tree's branch lengths at face value, so
      distortion of inner branch lengths (e.g., from reconstruction) also
      distorts the survival term, which neutral calibration does not
      correct; compare against the pendant estimator; and
    - nested branches share leaves, so the likelihood is a composite one.

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
        clades (see `alifestd_mask_trait_mdcs_asexual`). Losses are not yet
        supported.
    trait_names : Sequence[str], optional
        Labels for output rows; defaults to `trait_transitions`.
    neutral_traits : Sequence[int or str], default ()
        Indices or names of traits known to be neutral. Gain ratios are reported
        relative to their mean gain rate (leaving the focal trait out).
        Optional unless `calibrate` is set, which requires at least two.
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
        Number of equal-occupancy survival-time bins, over which each
        partition's survival term is averaged. Each bin stands in for its
        branches by their mean survival time, which understates the average
        survival term under a cost, so too few bins bias gain rate upward.
    num_bootstrap : int, default 0
        Number of bootstrap replicates, resampling branches with
        replacement and refitting, for standard errors,
        z-scores, and p-values. Calibration neighborhood sizes are held
        fixed across replicates.
    loss_rate : float, default 0.0
        Known per-unit-time trait loss rate.
    selection_starts : Sequence[float], default (0.0, -0.3, -1.8)
        Starting values for selection, in units of 1 / mean branch length;
        the best of the resulting fits is kept. See `tgs_fit`.
    max_growth_exponent : float, default 600.0
        Cap on carrier growth exponents, such as (selection - loss rate) *
        branch length, keeping optimizer trial steps finite. See `tgs_fit`.
    seed : int, optional
        Seed for bootstrap resampling.
    parallel_backend : str, optional
        joblib backend to run bootstrap replicates in parallel, e.g., "loky"
        (processes) or "threading".
    progress_wrap : Callable, default identity
        Pass tqdm or equivalent to display bootstrap progress.

    Returns
    -------
    pd.DataFrame
        One row per trait, with columns `trait`, `num_leaves` and
        `num_trait_present` (leaf branches that started without the trait,
        and those that gained it), `num_clades` and `num_clade_gains` (the
        same for inner branches), `gain_rate`, `selection`, `gain_ratio`,
        and `selection_lrt_naive`. With bootstrap replicates, also the
        pendant estimator's bootstrap columns.

    Raises
    ------
    ValueError
        If `calibrate` is set with fewer than two distinct neutral traits,
        if a `trait_transitions` column is missing or has values other than
        +1, -1, or 0, or gives states other than 0 and 1, if
        `trait_names` and `trait_transitions` differ in length, if
        `num_survival_time_bins` is less than 1, if
        `calibrate_size_partition_ratio` is not greater than 1, if
        `calibrate_size_partition_min_neutral_gains` is less than 1, if
        `calibrate_neighborhood_fractions` is empty or has a value outside
        (0, 1], or if `calibrate_num_partitions` is not a power of two.
    NotImplementedError
        If a `trait_transitions` column is not integer, or has losses
        (-1).

    Notes
    -----
    Unifurcations are bypassed. Branches with zero or undefined length
    (including roots) are excluded. Bootstrap replicates resample branches independently,
    ignoring correlation through shared ancestry, so standard errors are
    likely too small; nested branches share leaves, so the likelihood is a
    composite one.

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
        trait_names=trait_names,
        neutral_traits=neutral_traits,
        calibrate=calibrate,
        calibrate_neighborhood_fractions=calibrate_neighborhood_fractions,
        calibrate_num_partitions=calibrate_num_partitions,
    )
    branches = tgs_prepare_asexual(
        phylogeny_df,
        mutate,
        trait_transitions=trait_transitions,
        trait_names=trait_names,
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
        max_growth_exponent=max_growth_exponent,
        seed=seed,
        parallel_backend=parallel_backend,
        progress_wrap=progress_wrap,
    )
