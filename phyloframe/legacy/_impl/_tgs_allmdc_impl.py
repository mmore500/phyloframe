import functools
import logging
import typing

import numpy as np
import pandas as pd
import polars as pl

from ..._auxlib._identity import identity
from ..._auxlib._run_seeded_replicates import run_seeded_replicates
from ._tgs_find_allmdc_units import tgs_find_allmdc_units
from ._tgs_fit_allmdc_bootstrap_replicate import (
    tgs_fit_allmdc_bootstrap_replicate,
)
from ._tgs_fit_allmdc_traits import tgs_fit_allmdc_traits
from ._tgs_init_allmdc_results import tgs_init_allmdc_results
from ._tgs_make_allmdc_group import tgs_make_allmdc_group
from ._tgs_name_allmdc_shifts import tgs_name_allmdc_shifts
from ._tgs_summarize_allmdc_bootstrap import tgs_summarize_allmdc_bootstrap


def tgs_allmdc_impl(
    phylogeny_df: typing.Union[pd.DataFrame, pl.DataFrame],
    *,
    trait_names: typing.Sequence[str],
    neutral_traits: typing.Sequence[int],
    clade_size_groups_per_doubling: int,
    calibrate: bool,
    calibration_neighborhood_fractions: typing.Sequence[float],
    num_calibration_partitions: int,
    num_bootstrap: int,
    loss_rate: float,
    selection_starts: typing.Sequence[float],
    max_growth_exponent: float,
    max_shift_iterations: int,
    seed: typing.Optional[int],
    parallel_backend: typing.Optional[str],
    progress_wrap: typing.Callable,
) -> pd.DataFrame:
    """Implementation detail for
    `alifestd_tgs_allmdc_estimate_asexual` and
    `alifestd_tgs_allmdc_estimate_polars`.

    Here, tgs stands for trait gain/selection.

    Takes a phylogeny prepared by `tgs_prepare_asexual` or
    `tgs_prepare_polars`, with contiguous, topologically sorted ids and no
    unifurcations, and validated arguments (see `tgs_validate_args`).
    """
    origin_time = phylogeny_df["origin_time"].to_numpy()
    units = tgs_find_allmdc_units(
        ancestor_ids=phylogeny_df["ancestor_id"].to_numpy(),
        branch_lengths=phylogeny_df["__alifestd_tgs_branch_length"].to_numpy(),
        trait_masks=[
            phylogeny_df[f"__alifestd_tgs_trait{i}"].to_numpy()
            for i in range(len(trait_names))
        ],
        groups_per_doubling=clade_size_groups_per_doubling,
    )
    res = tgs_init_allmdc_results(trait_names=trait_names, units=units)
    branch_lengths = phylogeny_df["__alifestd_tgs_branch_length"].to_numpy()

    # (group index, group) for each group with branches to fit
    groups = []
    for j, nodes in enumerate(units.group_nodes):
        group = (
            tgs_make_allmdc_group(
                branch_lengths=branch_lengths[nodes],
                origin_times=origin_time[nodes],
                trait_units=units.trait_units[j],
                trait_gains=units.trait_gains[j],
                neutral_traits=neutral_traits,
                trait_names=trait_names,
                calibrate=calibrate,
                calibration_neighborhood_fractions=(
                    calibration_neighborhood_fractions
                ),
                num_calibration_partitions=num_calibration_partitions,
            )
            if len(nodes)
            else None
        )
        if group is None:
            logging.info(
                "- alifestd_tgs_allmdc_estimate_asexual: clades of "
                f"{units.group_bounds[j]} leaves have no branches to fit, or "
                "no neutral trait gains to calibrate against, so are not used",
            )
        else:
            groups.append((j, group))

    shift_columns = [None] + tgs_name_allmdc_shifts(units.group_bounds)
    fitted_columns = [shift_columns[j] for j, __ in groups]
    replicate_fits = []
    # without single leaves, there is no neutral clock to align groups to
    if groups and groups[0][0] == 0:
        finite_time = origin_time[np.isfinite(origin_time)]
        fit_kwargs = dict(
            groups=[group for __, group in groups],
            neutral_traits=neutral_traits,
            calibrate=calibrate,
            loss_rate=loss_rate,
            # a clade cannot take longer to grow out than the whole tree
            max_shift=float(np.ptp(finite_time)) or 1.0,
            num_traits=len(trait_names),
            selection_starts=selection_starts,
            max_growth_exponent=max_growth_exponent,
            max_shift_iterations=max_shift_iterations,
        )
        logging.info(
            "- alifestd_tgs_allmdc_estimate_asexual: fitting "
            f"{len(trait_names)} traits to {len(groups)} clade-size groups...",
        )
        point, shifts, params = tgs_fit_allmdc_traits(
            unit_weights=[np.ones(len(g.partitions)) for __, g in groups],
            starts=None,
            shifts_start=None,
            progress_wrap=progress_wrap,
            **fit_kwargs,
        )
        res[list(point.columns)] = point
        for column, shift in zip(fitted_columns[1:], shifts[1:]):
            res[column] = shift
        if num_bootstrap:
            replicate_fits = run_seeded_replicates(
                functools.partial(
                    tgs_fit_allmdc_bootstrap_replicate,
                    fit_kwargs=dict(
                        fit_kwargs,
                        starts=params,
                        shifts_start=shifts,
                        progress_wrap=identity,
                    ),
                ),
                num_replicates=num_bootstrap,
                seed=seed,
                parallel_backend=parallel_backend,
                progress_wrap=progress_wrap,
            )

    if not num_bootstrap:
        return res
    return tgs_summarize_allmdc_bootstrap(
        res=res,
        replicate_fits=replicate_fits,
        shift_columns=fitted_columns,
        num_bootstrap=num_bootstrap,
    )
