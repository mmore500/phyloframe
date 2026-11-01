import functools
import logging
import typing

import numpy as np
import pandas as pd

from ..._auxlib._run_seeded_replicates import run_seeded_replicates
from .._alifestd_mark_num_leaves_asexual import (
    _alifestd_mark_num_leaves_asexual_fast_path,
)
from ._tgs_count_traits import tgs_count_traits
from ._tgs_find_trait_mdcs import tgs_find_trait_mdcs
from ._tgs_fit_allmdc_bootstrap_replicate import (
    tgs_fit_allmdc_bootstrap_replicate,
)
from ._tgs_fit_allmdc_traits import TgsAllmdcGroup, tgs_fit_allmdc_traits
from ._tgs_init_results import tgs_init_results
from ._tgs_partition_leaves import tgs_partition_leaves
from ._tgs_select_calibration_neighborhoods import (
    tgs_select_calibration_neighborhoods,
)
from ._tgs_summarize_bootstrap import tgs_summarize_bootstrap
from ._tgs_validate_args import tgs_validate_args


def _shift_column(lower: int, upper: typing.Optional[int]) -> str:
    """Output column name for a clade-size group's shift."""
    if upper is None:
        return f"clade_size_shift_{lower}_plus"
    if upper == lower:
        return f"clade_size_shift_{lower}"
    return f"clade_size_shift_{lower}_{upper}"


def tgs_allmdc_impl(
    *,
    ancestor_ids: np.ndarray,
    origin_time: np.ndarray,
    branch_lengths: np.ndarray,
    trait_masks: typing.Sequence[np.ndarray],
    trait_names: typing.Sequence[str],
    neutral_traits: typing.Sequence[int],
    clade_size_groups: typing.Sequence[int],
    calibrate: bool,
    calibration_neighborhood_fractions: typing.Sequence[float],
    num_calibration_partitions: int,
    num_bootstrap: int,
    loss_rate: float,
    seed: typing.Optional[int],
    parallel_backend: typing.Optional[str],
    progress_wrap: typing.Callable,
) -> pd.DataFrame:
    """Implementation detail for
    `alifestd_estimate_trait_gain_selection_allmdc_asexual` and
    `alifestd_estimate_trait_gain_selection_allmdc_polars`.

    Here, tgs stands for trait gain/selection.

    Takes per-node arrays for a phylogeny with contiguous, topologically
    sorted ids and no unifurcations: ancestor ids, origin times, branch
    lengths (origin time deltas), and boolean trait masks.
    """
    neutral_traits = tgs_validate_args(
        num_traits=len(trait_masks),
        neutral_traits=neutral_traits,
        calibrate=calibrate,
        calibration_neighborhood_fractions=calibration_neighborhood_fractions,
        num_calibration_partitions=num_calibration_partitions,
    )
    lower_bounds = [int(bound) for bound in clade_size_groups]
    if any(bound < 2 for bound in lower_bounds) or any(
        a >= b for a, b in zip(lower_bounds, lower_bounds[1:])
    ):
        raise ValueError(
            "clade_size_groups must be strictly increasing, with each at "
            f"least 2, got {list(clade_size_groups)}",
        )
    num_traits = len(trait_masks)

    ancestor_ids = np.asarray(ancestor_ids, dtype=np.int64)
    origin_time = np.asarray(origin_time, dtype=np.float64)
    branch_lengths = np.asarray(branch_lengths, dtype=np.float64)
    # branches informative for fitting: those with positive length, so with
    # time to gain a trait on (excludes roots and nan)
    is_informative = branch_lengths > 0
    informative = np.flatnonzero(is_informative)
    num_leaves = _alifestd_mark_num_leaves_asexual_fast_path(ancestor_ids)
    # group 0 holds single leaves; group j > 0, clades of at least
    # lower_bounds[j - 1] leaves (and fewer than the next bound)
    unit_groups = np.searchsorted(
        lower_bounds, num_leaves[informative], side="right"
    )
    # (fewest, most) leaves of each group, with None for no upper bound
    group_bounds = [(1, 1)]
    group_bounds += [
        (lower, upper - 1)
        for lower, upper in zip(lower_bounds, lower_bounds[1:])
    ]
    group_bounds += [(lower_bounds[-1], None)] if lower_bounds else []
    num_groups = len(group_bounds)
    # position of each informative branch within its group
    group_positions = np.zeros(len(informative), dtype=np.int64)
    for j in range(num_groups):
        is_member = unit_groups == j
        group_positions[is_member] = np.arange(is_member.sum())

    trait_units = [[None] * num_traits for __ in range(num_groups)]
    trait_gains = [[None] * num_traits for __ in range(num_groups)]
    for trait, trait_mask in enumerate(trait_masks):
        units, gains = tgs_find_trait_mdcs(
            ancestor_ids=ancestor_ids,
            trait_mask=trait_mask,
            is_informative=is_informative,
        )
        for j in range(num_groups):
            trait_units[j][trait] = group_positions[
                units[unit_groups[units] == j]
            ]
            trait_gains[j][trait] = group_positions[
                gains[unit_groups[gains] == j]
            ]

    res = tgs_init_results(
        trait_names=trait_names,
        trait_leaves=trait_units[0],
        trait_carriers=trait_gains[0],
    )
    res.insert(
        3,
        "num_clades",
        [
            sum(len(trait_units[j][trait]) for j in range(1, num_groups))
            for trait in range(num_traits)
        ],
    )
    res.insert(
        4,
        "num_clade_gains",
        [
            sum(len(trait_gains[j][trait]) for j in range(1, num_groups))
            for trait in range(num_traits)
        ],
    )
    shift_columns = [_shift_column(*bounds) for bounds in group_bounds[1:]]
    for column in shift_columns:
        res[column] = np.nan

    groups = []
    for j in range(num_groups):
        members = informative[unit_groups == j]
        if not len(members):
            continue  # nothing to fit; its shift stays nan
        lengths, times = branch_lengths[members], origin_time[members]
        partitions, num_partitions, coords = tgs_partition_leaves(
            pendant_lengths=lengths,
            origin_times=times,
            max_partitions=num_calibration_partitions,
        )
        neighborhoods = None
        if calibrate:
            counts = tgs_count_traits(
                leaf_partitions=partitions,
                num_partitions=num_partitions,
                pendant_lengths=lengths,
                leaf_weights=np.ones(len(members)),
                trait_leaves=trait_units[j],
                trait_carriers=trait_gains[j],
            )
            if not counts.num_carriers[neutral_traits].any():
                logging.info(
                    "- alifestd_estimate_trait_gain_selection_allmdc_asexual: "
                    f"no neutral trait gains among clades of "
                    f"{group_bounds[j]} leaves to calibrate against, so "
                    "they are not used",
                )
                if j == 0:
                    break  # no single-leaf neutral clock to align against
                continue
            neighborhoods = tgs_select_calibration_neighborhoods(
                partition_coords=coords,
                neighborhood_fractions=calibration_neighborhood_fractions,
                counts=counts,
                neutral_traits=neutral_traits,
                trait_names=trait_names,
            )
        groups.append(
            (
                j,
                TgsAllmdcGroup(
                    partitions=partitions,
                    num_partitions=num_partitions,
                    branch_lengths=lengths,
                    trait_units=trait_units[j],
                    trait_gains=trait_gains[j],
                    calibration_neighborhoods=neighborhoods,
                ),
            )
        )

    replicate_fits = []
    if groups and groups[0][0] == 0:
        group_ids = [j for j, __ in groups]
        finite_time = origin_time[np.isfinite(origin_time)]
        fit_kwargs = dict(
            groups=[group for __, group in groups],
            neutral_traits=neutral_traits,
            calibrate=calibrate,
            loss_rate=loss_rate,
            # a clade cannot take longer to grow out than the whole tree
            max_shift=float(np.ptp(finite_time)) or 1.0,
            num_traits=num_traits,
        )
        logging.info(
            "- alifestd_estimate_trait_gain_selection_allmdc_asexual: "
            f"fitting {num_traits} traits to {len(informative)} branches...",
        )
        point, shifts, params = tgs_fit_allmdc_traits(
            unit_weights=[
                np.ones(len(group.partitions)) for __, group in groups
            ],
            **fit_kwargs,
        )
        res[list(point.columns)] = point
        for j, shift in zip(group_ids[1:], shifts[1:]):
            res[shift_columns[j - 1]] = shift
        if num_bootstrap:
            logging.info(
                "- alifestd_estimate_trait_gain_selection_allmdc_asexual: "
                f"fitting {num_bootstrap} bootstrap replicates...",
            )
            replicate_fits = run_seeded_replicates(
                functools.partial(
                    tgs_fit_allmdc_bootstrap_replicate,
                    fit_kwargs=dict(
                        fit_kwargs, starts=params, shifts_start=shifts
                    ),
                ),
                num_replicates=num_bootstrap,
                seed=seed,
                parallel_backend=parallel_backend,
                progress_wrap=progress_wrap,
            )

    if not num_bootstrap:
        return res
    if replicate_fits:
        fits = pd.concat(
            [fit for fit, __ in replicate_fits],
            keys=range(num_bootstrap),
            names=["replicate", "trait_index"],
        )
        replicate_shifts = np.array([shifts for __, shifts in replicate_fits])
    else:  # nothing to fit, so every replicate matches the empty fit
        fits = pd.concat(
            [res] * num_bootstrap,
            keys=range(num_bootstrap),
            names=["replicate", "trait_index"],
        )
        replicate_shifts = np.full((num_bootstrap, 0), np.nan)
    summary = tgs_summarize_bootstrap(point=res, replicate_fits=fits)
    res = pd.concat([res, summary], axis=1)
    for column in shift_columns:
        res[f"{column}_se"] = np.nan
    if replicate_fits:
        for k, j in enumerate(group_ids[1:], start=1):
            values = replicate_shifts[:, k]
            if np.isfinite(values).sum() >= 2:
                res[f"{shift_columns[j - 1]}_se"] = np.nanstd(values, ddof=1)
    return res
