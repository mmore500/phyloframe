import typing

import pandas as pd

from ._tgs_append_bootstrap_summary import tgs_append_bootstrap_summary
from ._tgs_find_trait_branches import tgs_find_trait_branches
from ._tgs_fit_and_bootstrap import tgs_fit_and_bootstrap
from ._tgs_init_results import tgs_init_results


def tgs_impl(
    branches: pd.DataFrame,
    *,
    trait_names: typing.Sequence[str],
    neutral_traits: typing.Sequence[int],
    calibrate: bool,
    calibrate_neighborhood_fractions: typing.Sequence[float],
    calibrate_num_partitions: int,
    num_bootstrap: int,
    loss_rate: float,
    min_leaf_origin_time: float,
    seed: typing.Optional[int],
    parallel_backend: typing.Optional[str],
    progress_wrap: typing.Callable,
) -> pd.DataFrame:
    """Implementation detail for
    `alifestd_tgs_pendantmdc_estimate_asexual` and
    `alifestd_tgs_pendantmdc_estimate_polars`.

    Here, tgs stands for trait gain/selection.

    Takes branches from `tgs_calc_branches` and validated arguments (see
    `tgs_validate_args`), and fits leaf pendant branches originating at or
    after `min_leaf_origin_time`.
    """
    leaves = branches[
        (branches["clade_size"] == 1)
        & (branches["origin_time"] >= min_leaf_origin_time)
    ]
    trait_leaves, trait_carriers = tgs_find_trait_branches(leaves)
    res = tgs_init_results(
        trait_names=trait_names,
        trait_leaves=trait_leaves,
        trait_carriers=trait_carriers,
    )
    replicate_fits = None
    if len(leaves):
        replicate_fits = tgs_fit_and_bootstrap(
            res=res,
            pendant_lengths=leaves["branch_length"].to_numpy(),
            origin_times=leaves["origin_time"].to_numpy(),
            trait_leaves=trait_leaves,
            trait_carriers=trait_carriers,
            trait_names=trait_names,
            neutral_traits=neutral_traits,
            calibrate=calibrate,
            calibrate_neighborhood_fractions=calibrate_neighborhood_fractions,
            calibrate_num_partitions=calibrate_num_partitions,
            num_bootstrap=num_bootstrap,
            loss_rate=loss_rate,
            seed=seed,
            parallel_backend=parallel_backend,
            progress_wrap=progress_wrap,
        )
    return tgs_append_bootstrap_summary(
        res, replicate_fits=replicate_fits, num_bootstrap=num_bootstrap
    )
