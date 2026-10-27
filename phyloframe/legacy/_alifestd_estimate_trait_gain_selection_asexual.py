import typing

import joblib
import numpy as np
import pandas as pd
from scipy import stats

from .._auxlib._calibrate_exposure import calibrate_exposure
from .._auxlib._fit_gain_selection_exposure import fit_gain_selection_exposure
from ._alifestd_mark_leaves import alifestd_mark_leaves
from ._alifestd_mark_origin_time_delta_asexual import (
    alifestd_mark_origin_time_delta_asexual,
)
from ._alifestd_mask_leaf_trait_inherited_asexual import (
    alifestd_mask_leaf_trait_inherited_asexual,
)
from ._alifestd_try_add_ancestor_id_col import alifestd_try_add_ancestor_id_col


def _alifestd_estimate_trait_gain_selection_asexual_fit(
    counts: typing.Sequence[typing.Tuple[np.ndarray, np.ndarray, np.ndarray]],
    block_weights: typing.Optional[np.ndarray],
    neutral_traits: typing.Sequence[int],
    calibrate: bool,
    exposure_cell_ids: np.ndarray,
    loss_rate: float,
) -> np.ndarray:
    """Implementation detail for
    `alifestd_estimate_trait_gain_selection_asexual`.

    Returns array of shape (num_traits, 4): gain rate, selection,
    likelihood-ratio statistic, and gain ratio versus neutral traits.
    """
    totals = [
        tuple(
            arr.sum(axis=0)
            if block_weights is None
            else np.tensordot(block_weights, arr, 1)
            for arr in trait_counts
        )
        for trait_counts in counts
    ]

    res = np.empty((len(totals), 4))
    for trait, (n, k, exposure_sum) in enumerate(totals):
        exposure = np.where(n > 0, exposure_sum / np.maximum(n, 1), 0.0)
        if calibrate:
            # leave focal trait out of its own calibration reference
            refs = [r for r in neutral_traits if r != trait]
            exposure = exposure * calibrate_exposure(
                sum(totals[r][1] for r in refs),
                sum(totals[r][2] for r in refs),
                exposure_cell_ids,
            )
        res[trait, :3] = fit_gain_selection_exposure(
            n, k, exposure, loss_rate=loss_rate
        )

    for trait in range(len(totals)):
        refs = [r for r in neutral_traits if r != trait]
        res[trait, 3] = res[trait, 0] / res[refs, 0].mean() if refs else np.nan

    return res


def _alifestd_estimate_trait_gain_selection_asexual_impl(
    origin_time: np.ndarray,
    length: np.ndarray,
    is_leaf: np.ndarray,
    trait_masks: typing.Sequence[np.ndarray],
    inherited_masks: typing.Sequence[np.ndarray],
    block_ids: typing.Optional[np.ndarray],
    trait_names: typing.Sequence[str],
    neutral_traits: typing.Sequence[int],
    calibrate: bool,
    num_bootstrap: int,
    loss_rate: float,
    min_leaf_origin_time: float,
    num_exposure_bins: int,
    num_calibration_exposure_cells: int,
    num_calibration_time_cells: int,
    seed: typing.Optional[int],
    parallel_backend: typing.Optional[str],
    progress_wrap: typing.Callable,
) -> pd.DataFrame:
    """Implementation detail for
    `alifestd_estimate_trait_gain_selection_asexual` and
    `alifestd_estimate_trait_gain_selection_polars`.

    Takes per-node arrays: origin time, pendant length, leaf mask, trait
    masks, inherited-trait masks (see
    `alifestd_mask_leaf_trait_inherited_asexual`), and bootstrap blocks.
    """
    origin_time = np.asarray(origin_time, dtype=np.float64)
    length = np.asarray(length, dtype=np.float64)
    is_used = (
        np.asarray(is_leaf, dtype=bool)
        & (length > 0)
        & (origin_time >= min_leaf_origin_time)
    )
    if not is_used.any():  # no informative leaves, so nothing to fit
        res = pd.DataFrame(
            {
                "trait": trait_names,
                "num_leaves": 0,
                "num_trait_present": 0,
                "gain_rate": 0.0,
                "selection": np.nan,
                "gain_ratio": np.nan,
                "selection_lrt_naive": np.nan,
            },
        )
        if num_bootstrap:
            for col in (
                "selection_se",
                "selection_z",
                "selection_p",
                "log_gain_ratio_se",
                "gain_ratio_z",
                "gain_ratio_p",
            ):
                res[col] = np.nan
        return res

    leaf_time, leaf_length = origin_time[is_used], length[is_used]
    leaf_block = (
        np.arange(is_used.sum())
        if block_ids is None
        else pd.factorize(np.asarray(block_ids)[is_used])[0]
    )
    num_blocks = leaf_block.max() + 1 if len(leaf_block) else 0

    # bin leaves by origin time and by pendant length
    time_width = (
        leaf_time.max() - origin_time.min()
    ) / num_calibration_time_cells
    time_bin = np.minimum(
        ((leaf_time - origin_time.min()) / time_width).astype(int),
        num_calibration_time_cells - 1,
    )
    length_bin = np.minimum(
        (
            leaf_length / (leaf_length.max() / num_exposure_bins * 1.0001)
        ).astype(int),
        num_exposure_bins - 1,
    )
    # group length bins into calibration cells holding similar leaf counts
    cuts = np.unique(
        np.quantile(
            length_bin,
            np.linspace(0, 1, num_calibration_exposure_cells + 1)[1:-1],
        ),
    )
    exposure_cell_ids = np.arange(num_calibration_time_cells)[:, None] * (
        len(cuts) + 1
    ) + np.searchsorted(cuts, np.arange(num_exposure_bins), side="right")

    # per-trait counts of leaves, trait carriers, and summed pendant length,
    # by block x time bin x length bin
    bin_index = (
        leaf_block * num_calibration_time_cells + time_bin
    ) * num_exposure_bins + length_bin
    shape = (num_blocks, num_calibration_time_cells, num_exposure_bins)
    counts, num_leaves, num_present = [], [], []
    for trait_mask, inherited_mask in zip(trait_masks, inherited_masks):
        present = np.asarray(trait_mask, dtype=bool)[is_used]
        keep = ~np.asarray(inherited_mask, dtype=bool)[is_used]
        counts.append(
            tuple(
                np.bincount(
                    bin_index[keep & mask],
                    weights=None if weights is None else weights[keep & mask],
                    minlength=np.prod(shape),
                )
                .reshape(shape)
                .astype(np.float64)
                for mask, weights in (
                    (keep, None),
                    (present, None),
                    (keep, leaf_length),
                )
            ),
        )
        num_leaves.append(int(keep.sum()))
        num_present.append(int((keep & present).sum()))

    fit_kwargs = dict(
        neutral_traits=neutral_traits,
        calibrate=calibrate,
        exposure_cell_ids=exposure_cell_ids,
        loss_rate=loss_rate,
    )
    point = _alifestd_estimate_trait_gain_selection_asexual_fit(
        counts, None, **fit_kwargs
    )
    res = pd.DataFrame(
        {
            "trait": trait_names,
            "num_leaves": num_leaves,
            "num_trait_present": num_present,
            "gain_rate": point[:, 0],
            "selection": point[:, 1],
            "gain_ratio": point[:, 3],
            "selection_lrt_naive": point[:, 2],
        },
    )

    if num_bootstrap:
        rngs = map(
            np.random.default_rng,
            np.random.SeedSequence(seed).spawn(num_bootstrap),
        )
        block_weights = (
            np.bincount(
                rng.integers(0, num_blocks, num_blocks), minlength=num_blocks
            ).astype(np.float64)
            for rng in rngs
        )
        fit = (joblib.delayed if parallel_backend else lambda x: x)(
            _alifestd_estimate_trait_gain_selection_asexual_fit
        )
        jobs = (fit(counts, w, **fit_kwargs) for w in block_weights)
        if parallel_backend:
            with joblib.parallel_backend(parallel_backend, n_jobs=-1):
                reps = joblib.Parallel()(progress_wrap(jobs))
        else:
            reps = [*progress_wrap(jobs)]
        reps = np.array(reps)

        res["selection_se"] = reps[:, :, 1].std(axis=0, ddof=1)
        res["selection_z"] = res["selection"] / res["selection_se"]
        res["selection_p"] = 2 * stats.norm.sf(np.abs(res["selection_z"]))
        res["log_gain_ratio_se"] = np.log(reps[:, :, 3]).std(axis=0, ddof=1)
        res["gain_ratio_z"] = (
            np.log(res["gain_ratio"]) / res["log_gain_ratio_se"]
        )
        res["gain_ratio_p"] = 2 * stats.norm.sf(np.abs(res["gain_ratio_z"]))

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
    num_exposure_bins: int = 800,
    num_calibration_exposure_cells: int = 12,
    num_calibration_time_cells: int = 8,
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

    which is linear in length at neutrality and saturates under a cost.

    Parameters
    ----------
    phylogeny_df : pd.DataFrame
        Phylogeny in alife standard format, with column `origin_time`.
    mutate : bool, default False
        If True, allows mutation of the input dataframe.
    trait_masks : Sequence[np.ndarray]
        One boolean array per trait marking nodes with the trait, aligned
        with `phylogeny_df.index`. Only leaf values are used.
    trait_names : Sequence[str], optional
        Labels for output rows; defaults to trait indices.
    neutral_traits : Sequence[int], default ()
        Indices of traits known to be neutral. Gain ratios are reported
        relative to their mean gain rate (leaving the focal trait out).
    calibrate : bool, default False
        If True, learn effective exposure for each (pendant length, leaf
        origin time) cell from `neutral_traits`, so that distortion of
        branch lengths shared with neutral traits (e.g., from tree
        reconstruction) cancels.
    block_ids : np.ndarray, optional
        Bootstrap block of each node, aligned with `phylogeny_df.index`;
        blocks should be approximately independent (e.g., spatial tiles or
        root lineages). Defaults to one block per leaf.
    num_bootstrap : int, default 0
        Number of block bootstrap replicates for standard errors, z-scores,
        and p-values.
    loss_rate : float, default 0.0
        Known per-unit-time trait loss rate.
    min_leaf_origin_time : float, default -inf
        Leaves originating earlier are excluded.
    num_exposure_bins : int, default 800
        Number of equal-width pendant length bins.
    num_calibration_exposure_cells : int, default 12
        Number of approximately equal-occupancy pendant length groups for
        calibration.
    num_calibration_time_cells : int, default 8
        Number of equal-width leaf origin time groups for calibration.
    seed : int, optional
        Seed for bootstrap resampling.
    parallel_backend : str, optional
        Pass "loky" to run bootstrap replicates in parallel with joblib.
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
        `gain_ratio` tests against 1).

    Notes
    -----
    Leaves with zero pendant length (including root leaves) are excluded.
    Under sparse sampling, selection estimates the change rate of a
    reconstructed carrier lineage's weight, roughly the difference in
    per-lineage net growth; magnitudes may be biased, but signs and
    rankings are informative. With few carriers, cost and loss are
    confounded, so `loss_rate` is fixed rather than fit.

    Input dataframe is not mutated by this operation unless `mutate` set
    True. If mutate set True, operation does not occur in place; still use
    return value to get transformed phylogeny dataframe.
    """
    num_traits = len(trait_masks)
    if trait_names is None:
        trait_names = list(map(str, range(num_traits)))
    if calibrate and len(neutral_traits) < 2:
        raise ValueError("calibration requires at least two neutral traits")

    if not mutate:
        phylogeny_df = phylogeny_df.copy()

    # stash per-node inputs as columns, so they follow any reordering
    trait_cols = [
        f"alifestd_estimate_trait_gain_selection_asexual_trait{i}"
        for i in range(num_traits)
    ]
    for col, mask in zip(trait_cols, trait_masks):
        phylogeny_df[col] = np.asarray(mask, dtype=bool)
    block_col = "alifestd_estimate_trait_gain_selection_asexual_block"
    phylogeny_df[block_col] = (
        np.arange(len(phylogeny_df))
        if block_ids is None
        else pd.factorize(np.asarray(block_ids))[0]
    )

    phylogeny_df = alifestd_try_add_ancestor_id_col(phylogeny_df, mutate=True)
    phylogeny_df = alifestd_mark_origin_time_delta_asexual(
        phylogeny_df,
        mutate=True,
        mark_as="alifestd_estimate_trait_gain_selection_asexual_length",
    )
    if "is_leaf" not in phylogeny_df.columns:
        phylogeny_df = alifestd_mark_leaves(phylogeny_df, mutate=True)

    inherited = []
    for col in trait_cols:
        phylogeny_df = alifestd_mask_leaf_trait_inherited_asexual(
            phylogeny_df,
            mutate=True,
            mark_as="alifestd_estimate_trait_gain_selection_asexual_inherited",
            trait_mask=phylogeny_df[col].to_numpy(),
        )
        inherited.append(
            phylogeny_df[
                "alifestd_estimate_trait_gain_selection_asexual_inherited"
            ].to_numpy()
        )

    res = _alifestd_estimate_trait_gain_selection_asexual_impl(
        origin_time=phylogeny_df["origin_time"].to_numpy(),
        length=phylogeny_df[
            "alifestd_estimate_trait_gain_selection_asexual_length"
        ].to_numpy(),
        is_leaf=phylogeny_df["is_leaf"].to_numpy(),
        trait_masks=[phylogeny_df[col].to_numpy() for col in trait_cols],
        inherited_masks=inherited,
        block_ids=phylogeny_df[block_col].to_numpy(),
        trait_names=trait_names,
        neutral_traits=neutral_traits,
        calibrate=calibrate,
        num_bootstrap=num_bootstrap,
        loss_rate=loss_rate,
        min_leaf_origin_time=min_leaf_origin_time,
        num_exposure_bins=num_exposure_bins,
        num_calibration_exposure_cells=num_calibration_exposure_cells,
        num_calibration_time_cells=num_calibration_time_cells,
        seed=seed,
        parallel_backend=parallel_backend,
        progress_wrap=progress_wrap,
    )

    phylogeny_df.drop(
        columns=[
            *trait_cols,
            block_col,
            "alifestd_estimate_trait_gain_selection_asexual_inherited",
        ],
        inplace=True,
    )
    return res
