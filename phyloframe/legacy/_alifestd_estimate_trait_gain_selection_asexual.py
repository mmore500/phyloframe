import functools
import logging
import operator
import typing
import warnings

import joblib
import numpy as np
import opytional as opyt
import pandas as pd
from scipy import stats

from .._auxlib._estimate_standardized_rate_ratios import (
    estimate_standardized_rate_ratios,
)
from .._auxlib._sample_block_bootstrap_counts import (
    sample_block_bootstrap_counts,
)
from .._auxlib._tgs_fit import TgsFitResult, tgs_fit
from .._auxlib._with_rng_state_context import with_rng_state_context
from ._alifestd_collapse_unifurcations import alifestd_collapse_unifurcations
from ._alifestd_mark_leaves import _alifestd_mark_leaves_asexual_fast_path
from ._alifestd_mask_leaf_trait_inherited_asexual import (
    _alifestd_mask_leaf_trait_inherited_asexual_coerce_mask,
    _alifestd_mask_leaf_trait_inherited_asexual_fast_path,
)
from ._alifestd_to_working_format import alifestd_to_working_format


def _tgs_bin_leaves(
    leaf_times: np.ndarray,
    pendant_lengths: np.ndarray,
    min_time: float,
    num_pendant_length_bins: int,
    num_calibration_pendant_length_groups: int,
    num_calibration_time_groups: int,
) -> typing.Tuple[np.ndarray, np.ndarray]:
    """Implementation detail for
    `alifestd_estimate_trait_gain_selection_asexual`.

    Here, tgs stands for trait gain/selection.

    Returns flat (time, length) bin of each leaf, and calibration group of
    each bin. Bins have equal width; calibration groups are time bins
    crossed with runs of length bins holding approximately equal leaf
    counts.
    """
    # digitize against interior edges only ([1:-1]), so that values at either
    # end land in the first or last bin rather than outside; with one bin,
    # there are no interior edges and every value lands in bin 0
    time_bins = np.digitize(
        leaf_times,
        np.linspace(
            min_time, leaf_times.max(), num_calibration_time_groups + 1
        )[1:-1],
    )
    length_bins = np.digitize(
        pendant_lengths,
        np.linspace(0, pendant_lengths.max(), num_pendant_length_bins + 1)[
            1:-1
        ],
    )
    length_quantiles = np.quantile(
        length_bins,
        np.linspace(0, 1, num_calibration_pendant_length_groups + 1)[1:-1],
    )
    length_groups = np.digitize(
        np.arange(num_pendant_length_bins), np.unique(length_quantiles)
    )
    # calibration group of each (time bin, length bin), in flat bin order
    group_ids = np.add.outer(
        np.arange(num_calibration_time_groups)
        * num_calibration_pendant_length_groups,
        length_groups,
    )
    return time_bins * num_pendant_length_bins + length_bins, group_ids.ravel()


def _tgs_fit_traits(
    leaf_bins: np.ndarray,
    pendant_lengths: np.ndarray,
    leaf_weights: np.ndarray,
    trait_leaves: typing.Sequence[np.ndarray],
    trait_carriers: typing.Sequence[np.ndarray],
    trait_names: typing.Sequence[str],
    neutral_traits: typing.Sequence[int],
    calibrate: bool,
    calibration_group_ids: np.ndarray,
    loss_rate: float,
    log_calibration: bool = False,
) -> pd.DataFrame:
    """Implementation detail for
    `alifestd_estimate_trait_gain_selection_asexual`.

    Here, tgs stands for trait gain/selection.

    Returns one row per trait, with columns of `TgsFitResult` plus
    `gain_ratio`, the gain rate relative to neutral traits.
    """
    bincount = functools.partial(
        np.bincount, minlength=len(calibration_group_ids)
    )
    weighted_lengths = leaf_weights * pendant_lengths
    counts = [  # per bin: leaves, trait carriers, summed pendant length
        (
            bincount(leaf_bins[leaves], leaf_weights[leaves]),
            bincount(leaf_bins[carriers], leaf_weights[carriers]),
            bincount(leaf_bins[leaves], weighted_lengths[leaves]),
        )
        for leaves, carriers in zip(trait_leaves, trait_carriers)
    ]
    # leave each trait out of its own neutral reference
    refs = [
        sorted(set(neutral_traits) - {trait}) for trait in range(len(counts))
    ]

    fits = []
    for trait, (n, k, pendant_length_sum) in enumerate(counts):
        pendant_length = np.divide(
            pendant_length_sum, n, out=np.zeros(n.shape), where=n > 0
        )
        if calibrate:
            ratios = estimate_standardized_rate_ratios(
                num_events=sum(counts[r][1] for r in refs[trait]),
                exposure=sum(counts[r][2] for r in refs[trait]),
                group_ids=calibration_group_ids,
            )
            if log_calibration and (n > 0).any():
                low, mid, high = np.quantile(ratios[n > 0], [0, 0.5, 1])
                logging.info(
                    "- alifestd_estimate_trait_gain_selection_asexual: "
                    f"calibrating trait {trait_names[trait]}, rescaling "
                    f"pendant lengths by {low:.3g} to {high:.3g} "
                    f"(median {mid:.3g})",
                )
            pendant_length *= ratios
        fits.append(tgs_fit(n, k, pendant_length, loss_rate=loss_rate))

    # normalize each gain rate by the mean gain rate of its neutral reference
    res = pd.DataFrame(fits, columns=TgsFitResult._fields)
    res["gain_ratio"] = [
        res["gain_rate"][trait] / res["gain_rate"][refs[trait]].mean()
        if refs[trait]
        else np.nan
        for trait in range(len(res))
    ]
    return res


def _tgs_bootstrap(
    fit_kwargs: dict,
    leaf_blocks: np.ndarray,
    num_bootstrap: int,
    parallel_backend: typing.Optional[str],
    progress_wrap: typing.Callable,
) -> typing.List[pd.DataFrame]:
    """Implementation detail for
    `alifestd_estimate_trait_gain_selection_asexual`.

    Here, tgs stands for trait gain/selection.

    Returns fits to block bootstrap resamples, one dataframe per replicate;
    see `_tgs_fit_traits`.
    """
    # each replicate weights leaves by how many times their block was drawn
    # (0, 1, 2, ..., not a binary mask), which is equivalent to copying the
    # leaves of each drawn block
    block_counts = sample_block_bootstrap_counts(
        leaf_blocks.max() + 1, num_bootstrap
    )
    jobs = (
        joblib.delayed(_tgs_fit_traits)(
            leaf_weights=counts[leaf_blocks].astype(np.float64), **fit_kwargs
        )
        for counts in block_counts
    )
    with joblib.parallel_backend(parallel_backend or "sequential", n_jobs=-1):
        return joblib.Parallel()(progress_wrap(jobs))


def _tgs_summarize_bootstrap(
    point: pd.DataFrame,
    reps: typing.Sequence[pd.DataFrame],
) -> pd.DataFrame:
    """Implementation detail for
    `alifestd_estimate_trait_gain_selection_asexual`.

    Here, tgs stands for trait gain/selection.

    Returns bootstrap standard errors, z-scores, and p-values, given point
    estimates and replicate fits.
    """
    # replicates resampling no carriers of a trait leave selection undefined
    # and gain ratio zero, so exclude them; gain ratio is also undefined
    # without neutral traits, which does not make a replicate degenerate
    # (selection is still estimable)
    selection_reps = np.array([rep["selection"] for rep in reps])
    with np.errstate(divide="ignore"):
        log_gain_ratio_reps = np.log([rep["gain_ratio"] for rep in reps])
    is_degenerate = ~np.isfinite(selection_reps)
    num_degenerate = is_degenerate.sum(axis=0)
    if num_degenerate.any():
        counts = ", ".join(
            f"{trait} ({count} of {len(reps)})"
            for trait, count in zip(point["trait"], num_degenerate)
            if count
        )
        warnings.warn(
            "bootstrap replicates that resampled no trait carriers were "
            f"excluded for traits {counts}, so their standard errors may be "
            "underestimated.",
        )

    def bootstrap_se(x: np.ndarray) -> np.ndarray:
        with warnings.catch_warnings():  # all-degenerate traits get nan
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanstd(
                np.where(is_degenerate, np.nan, x), axis=0, ddof=1
            )

    def two_sided_p(z: pd.Series) -> np.ndarray:
        return 2 * stats.norm.sf(np.abs(z))

    res = pd.DataFrame(index=point.index)
    res["selection_se"] = bootstrap_se(selection_reps)
    res["selection_z"] = point["selection"] / res["selection_se"]
    res["selection_p"] = two_sided_p(res["selection_z"])
    res["log_gain_ratio_se"] = bootstrap_se(log_gain_ratio_reps)
    res["gain_ratio_z"] = (
        np.log(point["gain_ratio"]) / res["log_gain_ratio_se"]
    )
    res["gain_ratio_p"] = two_sided_p(res["gain_ratio_z"])
    res["num_bootstrap_degenerate"] = num_degenerate
    return res


def _tgs_impl(
    ancestor_ids: np.ndarray,
    origin_time: np.ndarray,
    trait_masks: typing.Sequence[np.ndarray],
    block_ids: typing.Optional[np.ndarray],
    trait_names: typing.Sequence[str],
    neutral_traits: typing.Sequence[int],
    calibrate: bool,
    num_bootstrap: int,
    loss_rate: float,
    min_leaf_origin_time: float,
    num_pendant_length_bins: int,
    num_calibration_pendant_length_groups: int,
    num_calibration_time_groups: int,
    seed: typing.Optional[int],
    parallel_backend: typing.Optional[str],
    progress_wrap: typing.Callable,
) -> pd.DataFrame:
    """Implementation detail for
    `alifestd_estimate_trait_gain_selection_asexual` and
    `alifestd_estimate_trait_gain_selection_polars`.

    Here, tgs stands for trait gain/selection.

    Takes per-node arrays for a phylogeny with contiguous, topologically
    sorted ids and no unifurcations: ancestor ids, origin times, boolean
    trait masks, and bootstrap blocks.
    """
    for name, value in (
        ("num_pendant_length_bins", num_pendant_length_bins),
        (
            "num_calibration_pendant_length_groups",
            num_calibration_pendant_length_groups,
        ),
        ("num_calibration_time_groups", num_calibration_time_groups),
    ):
        if value < 1:
            raise ValueError(f"{name} must be at least 1, got {value}")

    neutral_traits = np.unique(
        np.arange(len(trait_masks))[list(map(operator.index, neutral_traits))]
    ).tolist()
    if calibrate and len(neutral_traits) < 2:
        raise ValueError("calibration requires at least two neutral traits")
    if neutral_traits:
        logging.info(
            "- alifestd_estimate_trait_gain_selection_asexual: "
            f"excluding each of {len(neutral_traits)} neutral traits from "
            "its own neutral reference",
        )

    ancestor_ids = np.asarray(ancestor_ids, dtype=np.int64)
    origin_time = np.asarray(origin_time, dtype=np.float64)
    length = origin_time - origin_time[ancestor_ids]
    is_used = (
        _alifestd_mark_leaves_asexual_fast_path(ancestor_ids)
        & (length > 0)  # excludes root leaves and nan
        & (origin_time >= min_leaf_origin_time)
    )

    # per trait, leaves that could have gained the trait on their pendant
    # branch, and those that did
    trait_leaves, trait_carriers = [], []
    for trait_mask in trait_masks:
        trait_mask = np.asarray(trait_mask, dtype=bool)
        inherited = _alifestd_mask_leaf_trait_inherited_asexual_fast_path(
            ancestor_ids, trait_mask
        )
        keep = ~inherited[is_used]
        trait_leaves.append(np.flatnonzero(keep))
        trait_carriers.append(np.flatnonzero(keep & trait_mask[is_used]))

    res = pd.DataFrame(
        {
            "trait": trait_names,
            "num_leaves": list(map(len, trait_leaves)),
            "num_trait_present": list(map(len, trait_carriers)),
            "gain_rate": 0.0,
            "selection": np.nan,
            "gain_ratio": np.nan,
            "selection_lrt_naive": np.nan,
        },
    )
    if not is_used.any():  # no informative leaves, so nothing to fit
        reps = [res] * num_bootstrap
    else:
        if block_ids is None:
            leaf_blocks = np.arange(is_used.sum())
        elif pd.isna(np.asarray(block_ids)[is_used]).any():
            raise ValueError(
                "block_ids must not be null at informative leaves",
            )
        else:
            leaf_blocks = pd.factorize(np.asarray(block_ids)[is_used])[0]

        leaf_bins, group_ids = _tgs_bin_leaves(
            origin_time[is_used],
            length[is_used],
            min_time=np.nanmin(origin_time),
            num_pendant_length_bins=num_pendant_length_bins,
            num_calibration_pendant_length_groups=num_calibration_pendant_length_groups,
            num_calibration_time_groups=num_calibration_time_groups,
        )
        fit_kwargs = dict(
            leaf_bins=leaf_bins,
            pendant_lengths=length[is_used],
            trait_leaves=trait_leaves,
            trait_carriers=trait_carriers,
            trait_names=trait_names,
            neutral_traits=neutral_traits,
            calibrate=calibrate,
            calibration_group_ids=group_ids,
            loss_rate=loss_rate,
        )
        logging.info(
            "- alifestd_estimate_trait_gain_selection_asexual: "
            f"fitting {len(trait_masks)} traits to {len(leaf_bins)} leaves...",
        )
        point = _tgs_fit_traits(
            leaf_weights=np.ones(len(leaf_bins)),
            log_calibration=True,
            **fit_kwargs,
        )
        res[list(point.columns)] = point
        if num_bootstrap:
            logging.info(
                "- alifestd_estimate_trait_gain_selection_asexual: "
                f"fitting {num_bootstrap} bootstrap replicates "
                f"over {leaf_blocks.max() + 1} blocks...",
            )
        bootstrap = opyt.apply_if_or_value(
            seed,
            lambda seed: with_rng_state_context(seed)(_tgs_bootstrap),
            _tgs_bootstrap,
        )
        reps = bootstrap(
            fit_kwargs,
            leaf_blocks,
            num_bootstrap=num_bootstrap,
            parallel_backend=parallel_backend,
            progress_wrap=progress_wrap,
        )

    if num_bootstrap:
        res = pd.concat([res, _tgs_summarize_bootstrap(res, reps)], axis=1)
    return res


def alifestd_estimate_trait_gain_selection_asexual(
    phylogeny_df: pd.DataFrame,
    mutate: bool = False,
    *,
    trait_masks: typing.Sequence[np.ndarray],
    trait_names: typing.Optional[typing.Sequence[str]] = None,
    neutral_traits: typing.Sequence[int] = (),
    calibrate: bool = False,
    block_ids: typing.Optional[np.ndarray] = None,
    num_bootstrap: int = 0,
    loss_rate: float = 0.0,
    min_leaf_origin_time: float = -np.inf,
    num_pendant_length_bins: int = 800,
    num_calibration_pendant_length_groups: int = 12,
    num_calibration_time_groups: int = 8,
    seed: typing.Optional[int] = None,
    parallel_backend: typing.Optional[str] = None,
    progress_wrap: typing.Callable = lambda x: x,
) -> pd.DataFrame:
    """Estimate per-trait gain rate and net selection from how often leaves
    carry each trait as a function of pendant branch length.

    Each leaf's pendant branch starts without the trait (unless the trait is
    shared across the parent's whole clade, in which case the leaf is
    dropped for that trait), so pendant length is an exposure window. Trait
    presence versus pendant length is fit to

        p = gain_rate * (exp(c * length) - 1) / c,  c = selection - loss_rate

    which is linear in length at neutrality and saturates under a cost. See
    `tgs_fit` for the model and its relation to BiSSE.

    Parameters
    ----------
    phylogeny_df : pd.DataFrame
        Phylogeny in alife standard format, with column `origin_time`.
    mutate : bool, default False
        If True, allows mutation of the input dataframe.
    trait_masks : Sequence[np.ndarray]
        One boolean array per trait marking nodes with the trait, aligned
        with `phylogeny_df.index`. Only leaf values are used. Null values
        are treated as trait absent, with a warning.
    trait_names : Sequence[str], optional
        Labels for output rows; defaults to trait indices.
    neutral_traits : Sequence[int], default ()
        Indices of traits known to be neutral. Gain ratios are reported
        relative to their mean gain rate (leaving the focal trait out).
    calibrate : bool, default False
        If True, rescale pendant lengths by the rate ratio of
        `neutral_traits` gains within each calibration group, i.e., each
        pairing of a pendant length group and a leaf origin time group (see
        `estimate_standardized_rate_ratios`), so that distortion of branch
        lengths shared with neutral traits (e.g., from tree reconstruction)
        cancels.
    block_ids : np.ndarray, optional
        Bootstrap block of each node, aligned with `phylogeny_df.index`;
        blocks should be approximately independent (e.g., spatial tiles or
        root lineages). Must not be null at informative leaves. Defaults to
        one block per leaf.
    num_bootstrap : int, default 0
        Number of block bootstrap replicates for standard errors, z-scores,
        and p-values.
    loss_rate : float, default 0.0
        Known per-unit-time trait loss rate.
    min_leaf_origin_time : float, default -inf
        Leaves originating earlier are excluded.
    num_pendant_length_bins : int, default 800
        Number of equal-width pendant length bins. Binning bounds fit cost
        independent of leaf count; the default keeps bin width under 0.2%
        of the longest pendant branch, so binning error is negligible.
    num_calibration_pendant_length_groups : int, default 12
        Number of approximately equal-occupancy pendant length groups for
        calibration.
    num_calibration_time_groups : int, default 8
        Number of equal-width leaf origin time groups for calibration.

        Calibration group defaults are heuristic: 12 x 8 = 96 calibration
        groups is coarse enough for each to accrue many neutral gains, yet
        resolves distortion that varies with pendant length and with depth.
        More groups reduce bias from within-group variation in distortion,
        at the cost of noisier calibration.
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
        resampling no carriers of the trait.

    Raises
    ------
    ValueError
        If `calibrate` is set with fewer than two distinct neutral traits,
        or if `block_ids` is null at an informative leaf.

    Notes
    -----
    Unifurcations are bypassed, so pendant branches span from leaves to
    their nearest branching ancestor. Leaves with zero or undefined pendant
    length (including root leaves) are excluded. Under sparse sampling,
    selection estimates the change rate of a reconstructed carrier
    lineage's weight, roughly the difference in per-lineage net growth;
    magnitudes may be biased, but signs and rankings are informative. With
    few carriers, cost and loss are confounded, so `loss_rate` is fixed
    rather than fit.

    Input dataframe is not mutated by this operation unless `mutate` set
    True. If mutate set True, operation does not occur in place; still use
    return value to get transformed phylogeny dataframe.

    See Also
    --------
    alifestd_estimate_trait_gain_selection_polars :
        Polars-based implementation.
    alifestd_mask_leaf_trait_inherited_asexual :
        Marks leaves dropped for each trait.
    """
    num_traits = len(trait_masks)
    if trait_names is None:
        trait_names = list(map(str, range(num_traits)))

    if not mutate:
        phylogeny_df = phylogeny_df.copy()

    # stash per-node inputs as columns, so they follow any reordering
    for i, (mask, name) in enumerate(zip(trait_masks, trait_names)):
        phylogeny_df[
            f"__alifestd_tgs_trait{i}"
        ] = _alifestd_mask_leaf_trait_inherited_asexual_coerce_mask(
            mask, name=f"trait mask {name}"
        )
    if block_ids is not None:
        phylogeny_df["__alifestd_tgs_block_ids"] = np.asarray(block_ids)

    phylogeny_df = alifestd_collapse_unifurcations(
        phylogeny_df, mutate=True, ignore_topological_sensitivity=True
    )
    phylogeny_df = alifestd_to_working_format(phylogeny_df, mutate=True)
    return _tgs_impl(
        ancestor_ids=phylogeny_df["ancestor_id"].to_numpy(),
        origin_time=phylogeny_df["origin_time"].to_numpy(),
        trait_masks=[
            phylogeny_df[f"__alifestd_tgs_trait{i}"].to_numpy()
            for i in range(num_traits)
        ],
        block_ids=(
            None
            if block_ids is None
            else phylogeny_df["__alifestd_tgs_block_ids"].to_numpy()
        ),
        trait_names=trait_names,
        neutral_traits=neutral_traits,
        calibrate=calibrate,
        num_bootstrap=num_bootstrap,
        loss_rate=loss_rate,
        min_leaf_origin_time=min_leaf_origin_time,
        num_pendant_length_bins=num_pendant_length_bins,
        num_calibration_pendant_length_groups=num_calibration_pendant_length_groups,
        num_calibration_time_groups=num_calibration_time_groups,
        seed=seed,
        parallel_backend=parallel_backend,
        progress_wrap=progress_wrap,
    )
