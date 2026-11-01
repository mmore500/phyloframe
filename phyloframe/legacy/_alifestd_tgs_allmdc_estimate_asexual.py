import typing

import numpy as np
import pandas as pd

from .._auxlib._identity import identity
from .._auxlib._warn_experimental import warn_experimental
from ._impl import tgs_allmdc_impl, tgs_prepare_asexual, tgs_validate_args


@warn_experimental
def alifestd_tgs_allmdc_estimate_asexual(
    phylogeny_df: pd.DataFrame,
    mutate: bool = False,
    *,
    trait_masks: typing.Sequence[np.ndarray],
    trait_names: typing.Optional[typing.Sequence[str]] = None,
    neutral_traits: typing.Sequence[typing.Union[int, str]] = (),
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
    clade_size_calibration_blocks: typing.Sequence[int] = (2, 3, 5, 9),
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
    trait_masks : Sequence[np.ndarray]
        One boolean array per trait marking nodes with the trait, aligned
        with `phylogeny_df.index`, with boolean dtype. Only leaf values are
        used.
    trait_names : Sequence[str], optional
        Labels for output rows; defaults to trait indices.
    neutral_traits : Sequence[int or str], default ()
        Indices or names of traits known to be neutral. Gain ratios are reported
        relative to their mean gain rate (leaving the focal trait out).
        Optional unless `calibrate` is set, which requires at least two.
    calibrate : bool, default False
        If True, rescale branch lengths by neutral trait gains, as in the
        pendant estimator, within neighborhoods of branches of the same
        clade-size block (see `clade_size_calibration_blocks`) similar in
        length and origin time. So, distortion of branch
        lengths shared with neutral traits, including any difference
        between inner and pendant branches, cancels.
    calibration_neighborhood_fractions : Sequence[float], default (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0)
        Candidate calibration neighborhood sizes, as fractions of each
        clade-size block's branches; see the pendant estimator.
    num_calibration_partitions : int, default 4096
        Maximum number of equal-occupancy partitions of each clade-size
        block's branches, by ranked length and origin time, for
        calibration and binning; a power of two.
    clade_size_calibration_blocks : Sequence[int], default (2, 3, 5, 9)
        Smallest clade size of each block after single leaves, so the
        default gives blocks of 1, 2, 3-4, 5-8, and 9+ leaves. Branches are
        partitioned and calibrated within their block, as neutral gains per
        unit length may differ with clade size; the model itself is the
        same for every block. () puts all branches in one block.
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
        if a trait mask's length differs from `phylogeny_df`'s, if
        `trait_names` and `trait_masks` differ in length, if
        `num_survival_time_bins` is less than 1, if
        `clade_size_calibration_blocks` is not strictly increasing from at
        least 2, if
        `calibration_neighborhood_fractions` is empty or has a value outside
        (0, 1], or if `num_calibration_partitions` is not a power of two.
    NotImplementedError
        If a trait mask does not have boolean dtype.

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
        trait_masks=trait_masks,
        trait_names=trait_names,
        neutral_traits=neutral_traits,
        calibrate=calibrate,
        calibration_neighborhood_fractions=calibration_neighborhood_fractions,
        num_calibration_partitions=num_calibration_partitions,
    )
    prepared = tgs_prepare_asexual(
        phylogeny_df,
        mutate,
        trait_masks=trait_masks,
        trait_names=trait_names,
    )
    return tgs_allmdc_impl(
        prepared,
        trait_names=trait_names,
        neutral_traits=neutral_traits,
        calibrate=calibrate,
        calibration_neighborhood_fractions=calibration_neighborhood_fractions,
        num_calibration_partitions=num_calibration_partitions,
        clade_size_calibration_blocks=clade_size_calibration_blocks,
        num_survival_time_bins=num_survival_time_bins,
        num_bootstrap=num_bootstrap,
        loss_rate=loss_rate,
        selection_starts=selection_starts,
        max_growth_exponent=max_growth_exponent,
        seed=seed,
        parallel_backend=parallel_backend,
        progress_wrap=progress_wrap,
    )
