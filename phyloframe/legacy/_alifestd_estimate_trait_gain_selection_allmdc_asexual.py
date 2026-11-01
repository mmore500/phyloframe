import typing

import numpy as np
import pandas as pd

from .._auxlib._warn_experimental import warn_experimental
from ._impl import tgs_allmdc_impl, tgs_prepare_asexual


@warn_experimental
def alifestd_estimate_trait_gain_selection_allmdc_asexual(
    phylogeny_df: pd.DataFrame,
    mutate: bool = False,
    *,
    trait_masks: typing.Sequence[np.ndarray],
    trait_names: typing.Optional[typing.Sequence[str]] = None,
    neutral_traits: typing.Sequence[int] = (),
    clade_size_groups: typing.Sequence[int] = (2, 3, 5, 9),
    calibrate: bool = False,
    calibration_neighborhood_fractions: typing.Sequence[float] = (
        0.005,
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
    ),
    num_calibration_partitions: int = 4096,
    num_bootstrap: int = 0,
    loss_rate: float = 0.0,
    seed: typing.Optional[int] = None,
    parallel_backend: typing.Optional[str] = None,
    progress_wrap: typing.Callable = lambda x: x,
) -> pd.DataFrame:
    """Estimate per-trait gain rate and net selection from every branch,
    pendant and inner, from how often each branch's clade carries the trait
    as a function of branch length and clade size.

    Here, an MDC is a mutationally defined clade: a maximal clade whose
    leaves all carry the trait, which, under parsimony without loss, gained
    it on its stem branch. This estimator uses MDCs of every size, single
    leaves included.

    Extends `alifestd_estimate_trait_gain_selection_pendantmdc_asexual`, which
    uses leaf pendant branches only, to inner branches, so that trait gains
    shared by a multi-leaf clade also inform the fit. Each branch whose
    parent's clade does not carry the trait at every leaf started without
    it, so branch length is an exposure window; the branch gained the trait
    (under parsimony, without loss) if its own clade carries it at every
    leaf. Gains are fit to

        p = gain_rate * (exp(c * length) - 1) / c * exp(c * shift),
        c = selection - loss_rate

    where `shift` is zero for leaves and, for a branch subtending a clade of
    several leaves, a time shared by all traits per clade-size group: the
    time a carrier clade of that size needs to grow out to its sampled
    leaves, during which a trait under cost must also persist. Gain rate and
    selection are fit per trait; shifts are fit jointly across traits.

    Notes on interpretation:

    - leaves are scored as in the pendant estimator, so with only leaf
      branches (e.g., a star tree), estimates match it;
    - a fixed shift per clade-size group approximates growth times that
      vary from clade to clade, and under strong cost may bias gain rate
      estimates low; compare against the pendant estimator; and
    - shifts are identified only by traits under selection with gains on
      inner branches; groups with no gains report nan shifts.

    Parameters
    ----------
    phylogeny_df : pd.DataFrame
        Phylogeny in alife standard format, with column `origin_time`.
    mutate : bool, default False
        If True, allows mutation of the input dataframe.
    trait_masks : Sequence[np.ndarray]
        One boolean array per trait marking nodes with the trait, aligned
        with `phylogeny_df.index`, with boolean dtype. Only leaf values are
        used.
    trait_names : Sequence[str], optional
        Labels for output rows; defaults to trait indices.
    neutral_traits : Sequence[int], default ()
        Indices of traits known to be neutral. Gain ratios are reported
        relative to their mean gain rate (leaving the focal trait out).
        Optional unless `calibrate` is set, which requires at least two.
    clade_size_groups : Sequence[int], default (2, 3, 5, 9)
        Smallest number of leaves in each clade-size group beyond single
        leaves, strictly increasing and at least 2; each group spans up to
        the next bound, and the last is open-ended. The default groups
        clades of 2, 3-4, 5-8, and 9 or more leaves.
    calibrate : bool, default False
        If True, rescale branch lengths by neutral trait gains, as in the
        pendant estimator, separately within each clade-size group, then
        rescale each group so that neutral traits gain at the single-leaf
        group's rate per unit length. So, distortion of branch lengths
        shared with neutral traits, including any difference between inner
        and pendant branches, cancels. Groups without neutral gains are not
        used.
    calibration_neighborhood_fractions : Sequence[float], default (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0)
        Candidate calibration neighborhood sizes, as fractions of each
        group's branches; see the pendant estimator.
    num_calibration_partitions : int, default 4096
        Maximum number of equal-occupancy partitions of each group's
        branches, by ranked length and origin time, for calibration and
        binning; a power of two.
    num_bootstrap : int, default 0
        Number of bootstrap replicates, resampling branches with
        replacement across all groups and refitting, for standard errors,
        z-scores, and p-values. Calibration neighborhood sizes are held
        fixed across replicates.
    loss_rate : float, default 0.0
        Known per-unit-time trait loss rate.
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
        `selection_lrt_naive`, and one `clade_size_shift_*` column per
        clade-size group (e.g., `clade_size_shift_2`,
        `clade_size_shift_3_4`, `clade_size_shift_9_plus`), the same in
        every row. With bootstrap replicates, also the pendant estimator's
        bootstrap columns and a `clade_size_shift_*_se` column per group.

    Raises
    ------
    ValueError
        If `calibrate` is set with fewer than two distinct neutral traits,
        if a trait mask's length differs from `phylogeny_df`'s, if
        `trait_names` and `trait_masks` differ in length, if
        `clade_size_groups` is not strictly increasing from at least 2, if
        `calibration_neighborhood_fractions` is empty or has a value outside
        (0, 1], or if `num_calibration_partitions` is not a power of two.
    NotImplementedError
        If a trait mask does not have boolean dtype.

    Notes
    -----
    Unifurcations are bypassed. Branches with zero or undefined length
    (including roots) are excluded. Shifts are bounded by the span of
    origin times. Bootstrap replicates resample branches independently,
    ignoring correlation through shared ancestry, so standard errors are
    likely too small; nested branches share leaves, so the likelihood is a
    composite one.

    Input dataframe is not mutated by this operation unless `mutate` set
    True. If mutate set True, operation does not occur in place; still use
    return value to get transformed phylogeny dataframe.

    See Also
    --------
    alifestd_estimate_trait_gain_selection_allmdc_polars :
        Polars-based implementation.
    alifestd_estimate_trait_gain_selection_pendantmdc_asexual :
        Leaf pendant branches only.
    """
    prepared = tgs_prepare_asexual(
        phylogeny_df,
        mutate,
        trait_masks=trait_masks,
        trait_names=trait_names,
    )
    return tgs_allmdc_impl(
        ancestor_ids=prepared.ancestor_ids,
        origin_time=prepared.origin_time,
        branch_lengths=prepared.branch_lengths,
        trait_masks=prepared.trait_masks,
        trait_names=prepared.trait_names,
        neutral_traits=neutral_traits,
        clade_size_groups=clade_size_groups,
        calibrate=calibrate,
        calibration_neighborhood_fractions=calibration_neighborhood_fractions,
        num_calibration_partitions=num_calibration_partitions,
        num_bootstrap=num_bootstrap,
        loss_rate=loss_rate,
        seed=seed,
        parallel_backend=parallel_backend,
        progress_wrap=progress_wrap,
    )
