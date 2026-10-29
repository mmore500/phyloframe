import functools
import logging
import operator
import typing
import warnings

import joblib
import numpy as np
import opytional as opyt
import pandas as pd
from scipy import sparse, stats
from sklearn import neighbors

from .._auxlib._calc_binomial_nll import calc_binomial_nll
from .._auxlib._calc_kdtree_partitions import calc_kdtree_partitions
from .._auxlib._calc_neighborhood_rate_ratios import (
    calc_neighborhood_rate_ratios,
)
from .._auxlib._sample_bootstrap_counts import sample_bootstrap_counts
from .._auxlib._tgs_fit import TgsFitResult, tgs_fit
from .._auxlib._with_rng_state_context import with_rng_state_context
from ._alifestd_collapse_unifurcations import alifestd_collapse_unifurcations
from ._alifestd_mark_leaves import _alifestd_mark_leaves_asexual_fast_path
from ._alifestd_mark_origin_time_delta_asexual import (
    alifestd_mark_origin_time_delta_asexual,
)
from ._alifestd_mask_leaf_trait_inherited_asexual import (
    _alifestd_mask_leaf_trait_inherited_asexual_fast_path,
)
from ._alifestd_to_working_format import alifestd_to_working_format

# per-trait counts in each partition: leaves, trait carriers, and summed
# pendant length; tgs stands for trait gain/selection
_TgsCounts = typing.Tuple[np.ndarray, np.ndarray, np.ndarray]


def _tgs_count_traits(
    *,
    leaf_partitions: np.ndarray,
    num_partitions: int,
    pendant_lengths: np.ndarray,
    leaf_weights: np.ndarray,
    trait_leaves: typing.Sequence[np.ndarray],
    trait_carriers: typing.Sequence[np.ndarray],
) -> typing.List[_TgsCounts]:
    """Implementation detail for
    `alifestd_estimate_trait_gain_selection_asexual`.

    Here, tgs stands for trait gain/selection.

    Returns, per trait, (weighted) counts of leaves and of trait carriers,
    and summed pendant length, in each partition.
    """
    bincount = functools.partial(np.bincount, minlength=num_partitions)
    weighted_lengths = leaf_weights * pendant_lengths
    return [
        (
            bincount(leaf_partitions[leaves], leaf_weights[leaves]),
            bincount(leaf_partitions[carriers], leaf_weights[carriers]),
            bincount(leaf_partitions[leaves], weighted_lengths[leaves]),
        )
        for leaves, carriers in zip(trait_leaves, trait_carriers)
    ]


def _tgs_calc_reference_rate_ratios(
    *,
    calibration_neighborhoods: sparse.spmatrix,
    counts: typing.Sequence[_TgsCounts],
    refs: typing.Sequence[int],
) -> np.ndarray:
    """Implementation detail for
    `alifestd_estimate_trait_gain_selection_asexual`.

    Here, tgs stands for trait gain/selection.

    Returns each partition's calibration neighborhood rate ratio of gains
    among the reference traits `refs`.
    """
    return calc_neighborhood_rate_ratios(
        neighborhoods=calibration_neighborhoods,
        num_events=sum(counts[ref][1] for ref in refs),
        exposure=sum(counts[ref][2] for ref in refs),
    )


def _tgs_calc_heldout_nll(
    *,
    calibration_neighborhoods: sparse.spmatrix,
    counts: typing.Sequence[_TgsCounts],
    neutral_traits: typing.Sequence[int],
    trait_names: typing.Sequence[str],
) -> float:
    """Implementation detail for `_tgs_select_calibration_neighborhoods`.

    Here, tgs stands for trait gain/selection.

    Returns summed negative log likelihood of each neutral trait's carriers
    under no selection, with pendant lengths calibrated against the other
    neutral traits only.
    """
    res = 0.0
    for trait in neutral_traits:
        n, k, pendant_length_sum = counts[trait]
        pendant_length = np.divide(
            pendant_length_sum, n, out=np.zeros(n.shape), where=n > 0
        ) * _tgs_calc_reference_rate_ratios(
            calibration_neighborhoods=calibration_neighborhoods,
            counts=counts,
            refs=sorted(set(neutral_traits) - {trait}),
        )
        exposure = (n * pendant_length).sum()
        if k.any() and exposure:
            gain_rate = k.sum() / exposure  # neutral fit, to first order
            res += calc_binomial_nll(n, k, gain_rate * pendant_length)[0]
        else:  # scores the same for every neighborhood size
            warnings.warn(
                f"neutral trait {trait_names[trait]} has no carriers or no "
                "calibrated exposure, so it does not inform the choice of "
                "calibration neighborhood size.",
            )
    return res


def _tgs_find_calibration_neighborhoods(
    *, partition_coords: np.ndarray, neighborhood_fraction: float
) -> sparse.spmatrix:
    """Implementation detail for `_tgs_select_calibration_neighborhoods`.

    Here, tgs stands for trait gain/selection.

    Returns partition-by-partition neighborhood membership matrix, whose row
    `i` marks partition `i`'s nearest partitions, including itself, together
    holding `neighborhood_fraction` of leaves.
    """
    num_partitions = len(partition_coords)
    k = np.clip(
        np.ceil(neighborhood_fraction * num_partitions), 1, num_partitions
    )
    # pass coordinates explicitly, so each partition is its own neighbor
    return (
        neighbors.NearestNeighbors(n_neighbors=int(k))
        .fit(partition_coords)
        .kneighbors_graph(partition_coords, mode="connectivity")
    )


def _tgs_select_calibration_neighborhoods(
    *,
    partition_coords: np.ndarray,
    neighborhood_fractions: typing.Sequence[float],
    counts: typing.Sequence[_TgsCounts],
    neutral_traits: typing.Sequence[int],
    trait_names: typing.Sequence[str],
) -> sparse.spmatrix:
    """Implementation detail for
    `alifestd_estimate_trait_gain_selection_asexual`.

    Here, tgs stands for trait gain/selection.

    Returns partition-by-partition neighborhood membership matrix for the
    neighborhood size, among `neighborhood_fractions`, that best predicts
    each held-out neutral trait's carriers when calibrating against the
    others.
    """
    best_nll, best_neighborhoods = np.inf, None
    for fraction in neighborhood_fractions:
        calibration_neighborhoods = _tgs_find_calibration_neighborhoods(
            partition_coords=partition_coords, neighborhood_fraction=fraction
        )
        nll = _tgs_calc_heldout_nll(
            calibration_neighborhoods=calibration_neighborhoods,
            counts=counts,
            neutral_traits=neutral_traits,
            trait_names=trait_names,
        )
        logging.info(
            "- alifestd_estimate_trait_gain_selection_asexual: "
            f"calibration neighborhoods of {fraction:.1%} of leaves give "
            f"held-out neutral negative log likelihood {nll:.1f}",
        )
        if nll < best_nll:
            best_nll, best_neighborhoods = nll, calibration_neighborhoods
    return best_neighborhoods


def _tgs_calibrate(
    *,
    partition_coords: np.ndarray,
    neighborhood_fractions: typing.Sequence[float],
    counts: typing.Sequence[_TgsCounts],
    neutral_traits: typing.Sequence[int],
    trait_names: typing.Sequence[str],
) -> sparse.spmatrix:
    """Implementation detail for
    `alifestd_estimate_trait_gain_selection_asexual`.

    Here, tgs stands for trait gain/selection.

    Returns calibration neighborhoods (see
    `_tgs_select_calibration_neighborhoods`), warning about any without
    neutral gains and logging each trait's rescaling of pendant lengths.
    """
    calibration_neighborhoods = _tgs_select_calibration_neighborhoods(
        partition_coords=partition_coords,
        neighborhood_fractions=neighborhood_fractions,
        counts=counts,
        neutral_traits=neutral_traits,
        trait_names=trait_names,
    )
    neutral_gains = sum(counts[trait][1] for trait in neutral_traits)
    num_empty = int((calibration_neighborhoods @ neutral_gains == 0).sum())
    if num_empty:
        warnings.warn(
            f"{num_empty} of {len(partition_coords)} calibration "
            "neighborhoods contain no neutral trait gains, so leaves there "
            "get zero effective pendant length; more neutral traits or "
            "leaves would help.",
        )

    for trait, (n, __, __) in enumerate(counts):
        if not (n > 0).any():
            logging.info(
                "- alifestd_estimate_trait_gain_selection_asexual: "
                f"not calibrating trait {trait_names[trait]}, which has no "
                "informative leaves",
            )
            continue
        ratios = _tgs_calc_reference_rate_ratios(
            calibration_neighborhoods=calibration_neighborhoods,
            counts=counts,
            refs=sorted(set(neutral_traits) - {trait}),
        )[n > 0]
        logging.info(
            "- alifestd_estimate_trait_gain_selection_asexual: "
            f"calibrating trait {trait_names[trait]}, rescaling pendant "
            f"lengths by {ratios.min():.3g} to {ratios.max():.3g} (median "
            f"{np.median(ratios):.3g})",
        )
    return calibration_neighborhoods


def _tgs_fit_traits(
    *,
    leaf_partitions: np.ndarray,
    num_partitions: int,
    pendant_lengths: np.ndarray,
    leaf_weights: np.ndarray,
    trait_leaves: typing.Sequence[np.ndarray],
    trait_carriers: typing.Sequence[np.ndarray],
    neutral_traits: typing.Sequence[int],
    calibration_neighborhoods: typing.Optional[sparse.spmatrix],
    loss_rate: float,
) -> pd.DataFrame:
    """Implementation detail for
    `alifestd_estimate_trait_gain_selection_asexual`.

    Here, tgs stands for trait gain/selection.

    Returns one row per trait, with columns of `TgsFitResult` plus
    `gain_ratio`, the gain rate relative to neutral traits. If
    `calibration_neighborhoods` is given, rescales each partition's pendant
    length by the rate ratio of neutral trait gains within its
    neighborhood.
    """
    counts = _tgs_count_traits(
        leaf_partitions=leaf_partitions,
        num_partitions=num_partitions,
        pendant_lengths=pendant_lengths,
        leaf_weights=leaf_weights,
        trait_leaves=trait_leaves,
        trait_carriers=trait_carriers,
    )
    # leave each trait out of its own neutral reference
    refs = [
        sorted(set(neutral_traits) - {trait}) for trait in range(len(counts))
    ]

    fits = []
    for trait, (n, k, pendant_length_sum) in enumerate(counts):
        pendant_length = np.divide(
            pendant_length_sum, n, out=np.zeros(n.shape), where=n > 0
        )
        if calibration_neighborhoods is not None:
            pendant_length *= _tgs_calc_reference_rate_ratios(
                calibration_neighborhoods=calibration_neighborhoods,
                counts=counts,
                refs=refs[trait],
            )
        # without carriers, skip fitting (and its warning), as callers report
        # such traits and replicates
        fits.append(
            tgs_fit(
                num_trials=n,
                num_successes=k,
                pendant_length=pendant_length,
                loss_rate=loss_rate,
            )
            if k.any()
            else TgsFitResult(
                gain_rate=0.0, selection=np.nan, selection_lrt_naive=np.nan
            )
        )

    # normalize each gain rate by the mean gain rate of its neutral reference;
    # undefined without neutral reference traits, or if none gained the trait
    res = pd.DataFrame(fits, columns=TgsFitResult._fields)
    ref_gain_rates = [
        res["gain_rate"][refs[trait]].mean() if len(refs[trait]) > 0 else 0.0
        for trait in range(len(res))
    ]
    res["gain_ratio"] = [
        gain_rate / ref_gain_rate if ref_gain_rate > 0 else np.nan
        for gain_rate, ref_gain_rate in zip(res["gain_rate"], ref_gain_rates)
    ]
    return res


def _tgs_fit_bootstrap_replicate(
    *, seed: typing.Optional[int], fit_kwargs: dict
) -> pd.DataFrame:
    """Implementation detail for `_tgs_bootstrap`.

    Here, tgs stands for trait gain/selection.

    Returns fit to one bootstrap resample of leaves; see `_tgs_fit_traits`.
    """
    sample = opyt.apply_if_or_value(
        seed,
        lambda seed: with_rng_state_context(seed)(sample_bootstrap_counts),
        sample_bootstrap_counts,
    )
    leaf_weights = sample(len(fit_kwargs["leaf_partitions"]))
    return _tgs_fit_traits(
        leaf_weights=leaf_weights.astype(np.float64), **fit_kwargs
    )


def _tgs_bootstrap(
    *,
    fit_kwargs: dict,
    num_bootstrap: int,
    seed: typing.Optional[int],
    parallel_backend: typing.Optional[str],
    progress_wrap: typing.Callable,
) -> pd.DataFrame:
    """Implementation detail for
    `alifestd_estimate_trait_gain_selection_asexual`.

    Here, tgs stands for trait gain/selection.

    Returns fits to bootstrap resamples of leaves, indexed by replicate and
    trait index; see `_tgs_fit_traits`.
    """
    # each replicate draws as many leaves as there are, with replacement, so
    # a leaf may be drawn zero, one, or several times; weighting each leaf by
    # its draw count in the fit is the same as refitting to the resampled
    # leaves, without building the resampled dataset; each replicate is
    # seeded separately, so that results do not depend on how or in what
    # order jobs are dispatched
    replicate_seeds = opyt.apply_if_or_value(
        seed,
        lambda seed: np.random.SeedSequence(seed).generate_state(
            num_bootstrap
        ),
        [None] * num_bootstrap,
    )
    jobs = (
        joblib.delayed(_tgs_fit_bootstrap_replicate)(
            seed=opyt.apply_if(replicate_seed, int), fit_kwargs=fit_kwargs
        )
        for replicate_seed in replicate_seeds
    )
    with joblib.parallel_backend(parallel_backend or "sequential", n_jobs=-1):
        replicate_fits = joblib.Parallel()(progress_wrap(jobs))
    return pd.concat(
        replicate_fits,
        keys=range(num_bootstrap),
        names=["replicate", "trait_index"],
    )


def _tgs_calc_two_sided_p(z: np.ndarray) -> np.ndarray:
    """Implementation detail for `_tgs_summarize_bootstrap`.

    Here, tgs stands for trait gain/selection.

    Returns two-sided p-value of each z-score under a standard normal.
    """
    return 2 * stats.norm.sf(np.abs(z))


def _tgs_summarize_bootstrap(
    *,
    point: pd.DataFrame,
    replicate_fits: pd.DataFrame,
) -> pd.DataFrame:
    """Implementation detail for
    `alifestd_estimate_trait_gain_selection_asexual`.

    Here, tgs stands for trait gain/selection.

    Returns bootstrap standard errors, z-scores, and p-values, given point
    estimates and replicate fits indexed by replicate and trait index.
    """
    # replicates resampling no carriers of a trait leave its selection
    # undefined, and those resampling no carriers of its neutral reference
    # leave its gain ratio undefined, so exclude them; gain ratio is also
    # undefined, by design, without neutral traits, which does not make a
    # replicate degenerate
    trait_index = replicate_fits.index.get_level_values("trait_index")
    selection = replicate_fits["selection"]
    gain_ratio = replicate_fits["gain_ratio"]
    is_selection_degenerate = ~np.isfinite(selection)
    is_gain_ratio_degenerate = is_selection_degenerate | (
        np.isfinite(point["gain_ratio"].to_numpy())[trait_index]
        & ~np.isfinite(gain_ratio)
    )
    num_degenerate = is_gain_ratio_degenerate.groupby(trait_index).sum()
    if num_degenerate.any():
        num_replicates = replicate_fits.index.get_level_values(0).nunique()
        counts = ", ".join(
            f"{trait} ({count} of {num_replicates})"
            for trait, count in zip(point["trait"], num_degenerate)
            if count
        )
        warnings.warn(
            "bootstrap replicates that resampled no carriers of a trait, or "
            f"of its neutral reference, were excluded for traits {counts}, "
            "so their standard errors may be underestimated.",
        )

    # standard deviation across replicates, per trait; nan with fewer than
    # two non-excluded replicates
    selection_se = (
        selection.mask(is_selection_degenerate)
        .groupby(trait_index)
        .std()
        .to_numpy()
    )
    log_gain_ratio_se = (
        np.log(gain_ratio.mask(is_gain_ratio_degenerate))
        .groupby(trait_index)
        .std()
        .to_numpy()
    )
    selection_z = point["selection"] / selection_se
    gain_ratio_z = np.log(point["gain_ratio"]) / log_gain_ratio_se
    return pd.DataFrame(
        {
            "selection_se": selection_se,
            "selection_z": selection_z,
            "selection_p": _tgs_calc_two_sided_p(selection_z),
            "log_gain_ratio_se": log_gain_ratio_se,
            "gain_ratio_z": gain_ratio_z,
            "gain_ratio_p": _tgs_calc_two_sided_p(gain_ratio_z),
            "num_bootstrap_degenerate": num_degenerate.to_numpy(),
        },
    )


def _tgs_find_trait_leaves(
    *,
    ancestor_ids: np.ndarray,
    trait_mask: np.ndarray,
    is_informative: np.ndarray,
) -> typing.Tuple[np.ndarray, np.ndarray]:
    """Implementation detail for
    `alifestd_estimate_trait_gain_selection_asexual`.

    Here, tgs stands for trait gain/selection.

    Returns positions, among informative leaves, of leaves whose trait
    state arose on their own pendant branch (i.e., whose trait is not
    shared by every leaf of their parent's clade), and of those among them
    that carry the trait.
    """
    trait_mask = np.asarray(trait_mask, dtype=bool)
    inherited = _alifestd_mask_leaf_trait_inherited_asexual_fast_path(
        ancestor_ids, trait_mask
    )
    is_own = ~inherited[is_informative]
    return (
        np.flatnonzero(is_own),
        np.flatnonzero(is_own & trait_mask[is_informative]),
    )


def _tgs_partition_leaves(
    *,
    pendant_lengths: np.ndarray,
    origin_times: np.ndarray,
    max_num_partitions: int,
) -> typing.Tuple[np.ndarray, int, np.ndarray]:
    """Implementation detail for
    `alifestd_estimate_trait_gain_selection_asexual`.

    Here, tgs stands for trait gain/selection.

    Returns partition of each leaf, number of partitions, and each
    partition's mean coordinates, grouping leaves into equal-occupancy
    partitions by ranked pendant length and origin time. Uses up to
    `max_num_partitions` partitions, a power of two, but keeps at least 4
    leaves in each.
    """
    num_leaves = len(pendant_lengths)
    leaf_ranks = np.column_stack(
        [
            stats.rankdata(x, method="ordinal") / num_leaves
            for x in (pendant_lengths, origin_times)
        ],
    )
    num_levels = int(
        max(0, min(np.log2(max_num_partitions), np.log2(num_leaves / 4)))
    )
    leaf_partitions = calc_kdtree_partitions(leaf_ranks, num_levels)
    num_partitions = 2**num_levels
    partition_sizes = np.bincount(leaf_partitions, minlength=num_partitions)
    partition_coords = np.column_stack(
        [
            np.bincount(leaf_partitions, x, minlength=num_partitions)
            / np.maximum(partition_sizes, 1)
            for x in leaf_ranks.T
        ],
    )
    return leaf_partitions, num_partitions, partition_coords


def _tgs_init_results(
    *,
    trait_names: typing.Sequence[str],
    trait_leaves: typing.Sequence[np.ndarray],
    trait_carriers: typing.Sequence[np.ndarray],
) -> pd.DataFrame:
    """Implementation detail for
    `alifestd_estimate_trait_gain_selection_asexual`.

    Here, tgs stands for trait gain/selection.

    Returns one row per trait, with leaf and carrier counts, and estimates
    for a trait with no fit (no gain, undefined selection).
    """
    return pd.DataFrame(
        (
            dict(
                trait=name,
                num_leaves=len(leaves),
                num_trait_present=len(carriers),
                gain_rate=0.0,
                selection=np.nan,
                gain_ratio=np.nan,
                selection_lrt_naive=np.nan,
            )
            for name, leaves, carriers in zip(
                trait_names, trait_leaves, trait_carriers
            )
        ),
        columns=[
            "trait",
            "num_leaves",
            "num_trait_present",
            "gain_rate",
            "selection",
            "gain_ratio",
            "selection_lrt_naive",
        ],
    )


def _tgs_fit(
    *,
    res: pd.DataFrame,
    pendant_lengths: np.ndarray,
    origin_times: np.ndarray,
    trait_leaves: typing.Sequence[np.ndarray],
    trait_carriers: typing.Sequence[np.ndarray],
    trait_names: typing.Sequence[str],
    neutral_traits: typing.Sequence[int],
    calibrate: bool,
    calibration_neighborhood_fractions: typing.Sequence[float],
    num_calibration_partitions: int,
    num_bootstrap: int,
    loss_rate: float,
    seed: typing.Optional[int],
    parallel_backend: typing.Optional[str],
    progress_wrap: typing.Callable,
) -> typing.Optional[pd.DataFrame]:
    """Implementation detail for
    `alifestd_estimate_trait_gain_selection_asexual`.

    Here, tgs stands for trait gain/selection.

    Fits each trait to informative leaves, given their pendant lengths and
    origin times, filling point estimates into `res` in place, and returns
    bootstrap replicate fits (None without bootstrap replicates).
    """
    leaf_partitions, num_partitions, partition_coords = _tgs_partition_leaves(
        pendant_lengths=pendant_lengths,
        origin_times=origin_times,
        max_num_partitions=num_calibration_partitions,
    )
    num_leaves = len(leaf_partitions)
    fit_kwargs = dict(
        leaf_partitions=leaf_partitions,
        num_partitions=num_partitions,
        pendant_lengths=pendant_lengths,
        trait_leaves=trait_leaves,
        trait_carriers=trait_carriers,
        neutral_traits=neutral_traits,
        calibration_neighborhoods=None,
        loss_rate=loss_rate,
    )
    if calibrate:
        fit_kwargs["calibration_neighborhoods"] = _tgs_calibrate(
            partition_coords=partition_coords,
            neighborhood_fractions=calibration_neighborhood_fractions,
            counts=_tgs_count_traits(
                leaf_partitions=leaf_partitions,
                num_partitions=num_partitions,
                pendant_lengths=pendant_lengths,
                leaf_weights=np.ones(num_leaves),
                trait_leaves=trait_leaves,
                trait_carriers=trait_carriers,
            ),
            neutral_traits=neutral_traits,
            trait_names=trait_names,
        )

    logging.info(
        "- alifestd_estimate_trait_gain_selection_asexual: "
        f"fitting {len(trait_leaves)} traits to {num_leaves} leaves...",
    )
    point = _tgs_fit_traits(leaf_weights=np.ones(num_leaves), **fit_kwargs)
    res[list(point.columns)] = point
    if not num_bootstrap:
        return None

    logging.info(
        "- alifestd_estimate_trait_gain_selection_asexual: "
        f"fitting {num_bootstrap} bootstrap replicates...",
    )
    return _tgs_bootstrap(
        fit_kwargs=fit_kwargs,
        num_bootstrap=num_bootstrap,
        seed=seed,
        parallel_backend=parallel_backend,
        progress_wrap=progress_wrap,
    )


def _tgs_impl(
    *,
    ancestor_ids: np.ndarray,
    origin_time: np.ndarray,
    pendant_lengths: np.ndarray,
    trait_masks: typing.Sequence[np.ndarray],
    trait_names: typing.Sequence[str],
    neutral_traits: typing.Sequence[int],
    calibrate: bool,
    calibration_neighborhood_fractions: typing.Sequence[float],
    num_calibration_partitions: int,
    num_bootstrap: int,
    loss_rate: float,
    min_leaf_origin_time: float,
    seed: typing.Optional[int],
    parallel_backend: typing.Optional[str],
    progress_wrap: typing.Callable,
) -> pd.DataFrame:
    """Implementation detail for
    `alifestd_estimate_trait_gain_selection_asexual` and
    `alifestd_estimate_trait_gain_selection_polars`.

    Here, tgs stands for trait gain/selection.

    Takes per-node arrays for a phylogeny with contiguous, topologically
    sorted ids and no unifurcations: ancestor ids, origin times, pendant
    branch lengths (origin time deltas), and boolean trait masks.
    """
    neutral_traits = np.unique(
        np.arange(len(trait_masks))[list(map(operator.index, neutral_traits))]
    ).tolist()
    if calibrate and len(neutral_traits) < 2:
        raise ValueError("calibration requires at least two neutral traits")
    if not calibration_neighborhood_fractions or not all(
        0 < fraction <= 1 for fraction in calibration_neighborhood_fractions
    ):
        raise ValueError(
            "calibration_neighborhood_fractions must be nonempty, with each "
            "in (0, 1]",
        )
    if num_calibration_partitions < 1 or (
        num_calibration_partitions & (num_calibration_partitions - 1)
    ):
        raise ValueError(
            "num_calibration_partitions must be a power of two, got "
            f"{num_calibration_partitions}",
        )

    ancestor_ids = np.asarray(ancestor_ids, dtype=np.int64)
    origin_time = np.asarray(origin_time, dtype=np.float64)
    pendant_lengths = np.asarray(pendant_lengths, dtype=np.float64)
    # leaves informative for fitting: those with a pendant branch to gain a
    # trait on, originating late enough to include
    is_informative = (
        _alifestd_mark_leaves_asexual_fast_path(ancestor_ids)
        & (pendant_lengths > 0)  # excludes root leaves and nan
        & (origin_time >= min_leaf_origin_time)
    )
    # for each trait, informative leaves whose trait state reflects their own
    # pendant branch rather than inheritance, and the carriers among them
    trait_leaves, trait_carriers = [], []
    for trait_mask in trait_masks:
        leaves, carriers = _tgs_find_trait_leaves(
            ancestor_ids=ancestor_ids,
            trait_mask=trait_mask,
            is_informative=is_informative,
        )
        trait_leaves.append(leaves)
        trait_carriers.append(carriers)
    res = _tgs_init_results(
        trait_names=trait_names,
        trait_leaves=trait_leaves,
        trait_carriers=trait_carriers,
    )
    if is_informative.any():
        replicate_fits = _tgs_fit(
            res=res,
            pendant_lengths=pendant_lengths[is_informative],
            origin_times=origin_time[is_informative],
            trait_leaves=trait_leaves,
            trait_carriers=trait_carriers,
            trait_names=trait_names,
            neutral_traits=neutral_traits,
            calibrate=calibrate,
            calibration_neighborhood_fractions=calibration_neighborhood_fractions,
            num_calibration_partitions=num_calibration_partitions,
            num_bootstrap=num_bootstrap,
            loss_rate=loss_rate,
            seed=seed,
            parallel_backend=parallel_backend,
            progress_wrap=progress_wrap,
        )
    else:  # nothing to fit, so every replicate matches the empty fit
        replicate_fits = pd.concat(
            [res] * num_bootstrap,
            keys=range(num_bootstrap),
            names=["replicate", "trait_index"],
        )

    if not num_bootstrap:
        return res
    return pd.concat(
        [
            res,
            _tgs_summarize_bootstrap(point=res, replicate_fits=replicate_fits),
        ],
        axis=1,
    )


def alifestd_estimate_trait_gain_selection_asexual(
    phylogeny_df: pd.DataFrame,
    mutate: bool = False,
    *,
    trait_masks: typing.Sequence[np.ndarray],
    trait_names: typing.Optional[typing.Sequence[str]] = None,
    neutral_traits: typing.Sequence[int] = (),
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
    min_leaf_origin_time: float = -np.inf,
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
        with `phylogeny_df.index`, with boolean dtype. Only leaf values are
        used.
    trait_names : Sequence[str], optional
        Labels for output rows; defaults to trait indices.
    neutral_traits : Sequence[int], default ()
        Indices of traits known to be neutral. Gain ratios are reported
        relative to their mean gain rate (leaving the focal trait out).
        Optional unless `calibrate` is set, which requires at least two.
        Without neutral traits, `gain_ratio` and its statistics are nan;
        gain rate and selection are still estimated.
    calibrate : bool, default False
        If True, rescale each leaf's pendant length by the ratio of observed
        to expected `neutral_traits` gains among its nearest leaves in
        pendant length and origin time (each ranked), so that distortion of
        branch lengths shared with neutral traits (e.g., from tree
        reconstruction) cancels. The neighborhood size is chosen, among
        `calibration_neighborhood_fractions`, as the one that best predicts
        each held-out neutral trait's carriers when calibrating against the
        others. Assumes loss is negligible over a pendant branch.
    calibration_neighborhood_fractions : Sequence[float], default (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0)
        Candidate calibration neighborhood sizes, as fractions of leaves.
        With `calibrate`, the candidate whose calibration best predicts
        held-out neutral traits is used. Larger neighborhoods smooth more;
        smaller ones resolve finer distortion but are noisier. A fraction of
        1.0 makes one neighborhood of all leaves, i.e., no calibration.
    num_calibration_partitions : int, default 4096
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
        if a trait mask's length differs from `phylogeny_df`'s, if
        `trait_names` and `trait_masks` differ in length, if
        `calibration_neighborhood_fractions` is empty or has a value outside
        (0, 1], or if `num_calibration_partitions` is not a power of two.
    NotImplementedError
        If a trait mask does not have boolean dtype.

    Notes
    -----
    Unifurcations are bypassed, so pendant branches span from leaves to
    their nearest branching ancestor. Leaves with zero or undefined pendant
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
    alifestd_estimate_trait_gain_selection_polars :
        Polars-based implementation.
    alifestd_mask_leaf_trait_inherited_asexual :
        Marks leaves dropped for each trait.
    """
    num_traits = len(trait_masks)
    if trait_names is None:
        trait_names = list(map(str, range(num_traits)))
    if len(trait_names) != num_traits:
        raise ValueError(
            f"got {len(trait_names)} trait names for {num_traits} trait masks",
        )

    if not mutate:
        phylogeny_df = phylogeny_df.copy()

    trait_masks = list(map(np.asarray, trait_masks))
    for mask, name in zip(trait_masks, trait_names):
        if mask.dtype != np.bool_:
            raise NotImplementedError(
                f"trait mask {name} must have boolean dtype, got {mask.dtype}",
            )
        if len(mask) != len(phylogeny_df):
            raise ValueError(
                f"trait mask {name} has length {len(mask)}, but "
                f"phylogeny_df has {len(phylogeny_df)} rows",
            )
    # track each node's original row, as unifurcation collapse and working
    # format conversion drop and reorder rows
    phylogeny_df["__alifestd_tgs_row"] = np.arange(len(phylogeny_df))
    phylogeny_df = alifestd_collapse_unifurcations(
        phylogeny_df, mutate=True, ignore_topological_sensitivity=True
    )
    phylogeny_df = alifestd_to_working_format(phylogeny_df, mutate=True)
    phylogeny_df = alifestd_mark_origin_time_delta_asexual(
        phylogeny_df, mutate=True, mark_as="__alifestd_tgs_pendant_length"
    )
    rows = phylogeny_df["__alifestd_tgs_row"].to_numpy()
    return _tgs_impl(
        ancestor_ids=phylogeny_df["ancestor_id"].to_numpy(),
        origin_time=phylogeny_df["origin_time"].to_numpy(),
        pendant_lengths=phylogeny_df[
            "__alifestd_tgs_pendant_length"
        ].to_numpy(),
        trait_masks=[mask[rows] for mask in trait_masks],
        trait_names=trait_names,
        neutral_traits=neutral_traits,
        calibrate=calibrate,
        calibration_neighborhood_fractions=calibration_neighborhood_fractions,
        num_calibration_partitions=num_calibration_partitions,
        num_bootstrap=num_bootstrap,
        loss_rate=loss_rate,
        min_leaf_origin_time=min_leaf_origin_time,
        seed=seed,
        parallel_backend=parallel_backend,
        progress_wrap=progress_wrap,
    )
