import typing

import numpy as np
import pandas as pd

from .._auxlib._identity import identity
from .._auxlib._warn_experimental import warn_experimental
from ._impl import tgs_impl, tgs_prepare_asexual, tgs_validate_args


@warn_experimental
def alifestd_tgs_pendantmdc_estimate_asexual(
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
    num_bootstrap: int = 0,
    loss_rate: float = 0.0,
    min_leaf_origin_time: float = -np.inf,
    seed: typing.Optional[int] = None,
    parallel_backend: typing.Optional[str] = None,
    progress_wrap: typing.Callable = identity,
) -> pd.DataFrame:
    """Estimate per-trait gain rate and net selection from leaf pendant
    branches only, from how the fraction of leaves carrying each trait
    varies with pendant branch length.

    Here, an MDC is a mutationally defined clade: a maximal clade whose
    leaves all carry the trait, which, under parsimony without loss, gained
    it on its stem branch. This estimator uses single-leaf MDCs only (gains
    on pendant branches); see
    `alifestd_tgs_allmdc_estimate_asexual` for MDCs of
    every size.

    Each leaf's pendant branch starts without the trait (unless the trait is
    shared across the parent's whole clade, in which case the leaf is
    dropped for that trait), so pendant length is an exposure window. Trait
    presence versus pendant length is fit to

        p = gain_rate * (exp(c * length) - 1) / c,  c = selection - loss_rate

    which is linear in length at neutrality and saturates under a cost.

    Uses pendant branches only, in that:

    - only gains on a leaf's own pendant branch are counted; leaves whose
      trait is shared by their parent's whole clade (i.e., gained on an
      inner branch) are dropped for that trait, so trait-defined clades
      contribute nothing;
    - inner branches contribute no exposure, and branching within carrier
      clades (the rate at which they diversify) is not modeled; and
    - each leaf is scored independently, given its pendant length, rather
      than through a full tree likelihood (e.g., BiSSE).

    Calibration also uses pendant branches only, comparing neutral traits'
    pendant gains across leaves of similar pendant length and origin time.

    See `tgs_fit` for the model and its relation to BiSSE.

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
        Used to calibrate pendant lengths (with `calibrate`), which requires
        at least two, and as the reference for gain ratios (relative to
        their mean gain rate, leaving the focal trait out).
        Without neutral traits, `gain_ratio` and its statistics are nan;
        gain rate and selection are still estimated.
    calibrate : bool, default False
        If True, rescale each leaf's pendant length by the ratio of observed
        to expected `neutral_traits` gains among its nearest leaves in
        pendant length and origin time (each ranked), so that distortion of
        branch lengths shared with neutral traits (e.g., from tree
        reconstruction) cancels. The neighborhood size is chosen, among
        `calibrate_neighborhood_fractions`, as the one that best predicts
        each held-out neutral trait's carriers when calibrating against the
        others. Assumes loss is negligible over a pendant branch.
    calibrate_neighborhood_fractions : Sequence[float], default (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0)
        Candidate calibration neighborhood sizes, as fractions of leaves.
        With `calibrate`, the candidate whose calibration best predicts
        held-out neutral traits is used. Larger neighborhoods smooth more;
        smaller ones resolve finer distortion but are noisier. A fraction of
        1.0 makes one neighborhood of all leaves, i.e., no calibration.
    calibrate_num_partitions : int, default 4096
        Number of equal-occupancy partitions of leaves, by ranked pendant
        length and origin time, among which calibration neighborhoods are
        formed; a power of two. Also bins leaves for fitting, so that fit
        cost is bounded independent of leaf count. Fewer are used if needed
        to keep at least 4 leaves in each.
    num_bootstrap : int, default 0
        Number of bootstrap replicates, resampling leaves with replacement
        and refitting, for standard errors, z-scores, and p-values. Each
        replicate recomputes calibration rate ratios from its resample, but
        holds fixed the calibration neighborhood size chosen on the full
        data, so standard errors omit uncertainty in that choice.
    loss_rate : float, default 0.0
        Known per-unit-time trait loss rate.
    min_leaf_origin_time : float, default -inf
        Leaves originating earlier are excluded.
    seed : int, optional
        Seed for bootstrap resampling.
    parallel_backend : str, optional
        joblib backend to run bootstrap replicates in parallel, e.g., "loky"
        (processes) or "threading". Replicate fits are mostly GIL-bound
        Python (scipy's optimizer loop), so "loky" is typically faster.
    progress_wrap : Callable, default identity
        Pass tqdm or equivalent to display bootstrap progress.

    Returns
    -------
    pd.DataFrame
        One row per trait, with columns `trait`, `num_leaves`,
        `num_trait_present`, `gain_rate`, `selection`, `gain_ratio`, and
        `selection_lrt_naive`. With bootstrap replicates, also
        `selection_se`, `selection_z`, `selection_p`, `log_gain_ratio_se`,
        `gain_ratio_z`, and `gain_ratio_p` (two-sided normal approximation;
        `gain_ratio` tests against 1), as well as
        `num_bootstrap_degenerate`, the number of replicates excluded for
        resampling no carriers of the trait, or of its neutral reference.

    Raises
    ------
    ValueError
        If `calibrate` is set with fewer than two distinct neutral traits,
        if a `trait_transitions` column is missing or has values other than
        +1, -1, or 0, or gives states other than 0 and 1, if a
        `neutral_traits` name is not among `trait_transitions`, if
        `calibrate_neighborhood_fractions` is empty or has a value outside
        (0, 1], or if `calibrate_num_partitions` is not a power of two.
    NotImplementedError
        If a `trait_transitions` column is not integer, or has loss
        events, i.e., -1 values.

    Notes
    -----
    Unifurcations are bypassed, so pendant branches span from leaves to
    their stem node, the nearest ancestor with at least two children.
    Leaves with zero or undefined pendant
    length (including root leaves) are excluded. Under sparse sampling,
    selection estimates the change rate of a reconstructed carrier
    lineage's weight, roughly the difference in per-lineage net growth;
    magnitudes may be biased, but signs and rankings are informative. With
    few carriers, cost and loss are confounded, so `loss_rate` is fixed
    rather than fit. Bootstrap replicates resample leaves independently,
    ignoring correlation through shared ancestry.

    Input dataframe is not mutated by this operation unless `mutate` set
    True. If mutate set True, operation does not occur in place; still use
    return value to get transformed phylogeny dataframe.

    See Also
    --------
    alifestd_tgs_pendantmdc_estimate_polars :
        Polars-based implementation.
    alifestd_mask_trait_singletons_asexual :
        Marks leaves dropped for each trait.
    """
    trait_names, neutral_traits = tgs_validate_args(
        trait_transitions=trait_transitions,
        neutral_traits=neutral_traits,
        calibrate=calibrate,
        calibrate_neighborhood_fractions=calibrate_neighborhood_fractions,
        calibrate_num_partitions=calibrate_num_partitions,
    )
    exposure_df = tgs_prepare_asexual(
        phylogeny_df,
        mutate,
        trait_transitions=trait_transitions,
    )
    return tgs_impl(
        exposure_df,
        trait_names=trait_names,
        neutral_traits=neutral_traits,
        calibrate=calibrate,
        calibrate_neighborhood_fractions=calibrate_neighborhood_fractions,
        calibrate_num_partitions=calibrate_num_partitions,
        num_bootstrap=num_bootstrap,
        loss_rate=loss_rate,
        min_leaf_origin_time=min_leaf_origin_time,
        seed=seed,
        parallel_backend=parallel_backend,
        progress_wrap=progress_wrap,
    )
