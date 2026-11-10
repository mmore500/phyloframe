import gc
import logging
import typing
import warnings

import numpy as np
import polars as pl
import polars.selectors as cs

from ..._auxlib._log_memory_usage import log_memory_usage
from .._alifestd_count_unifurcating_roots_polars import (
    alifestd_count_unifurcating_roots_polars,
)
from .._alifestd_count_unifurcations_polars import (
    alifestd_count_unifurcations_polars,
)
from .._alifestd_has_contiguous_ids_polars import (
    alifestd_has_contiguous_ids_polars,
)
from .._alifestd_is_topologically_sorted_polars import (
    alifestd_is_topologically_sorted_polars,
)
from .._alifestd_mark_lineage_cumsum_asexual import (
    _alifestd_mark_lineage_cumsum_asexual_fast_path,
)
from .._alifestd_mark_max_descendant_origin_time_asexual import (
    _alifestd_mark_max_descendant_origin_time_asexual_fast_path,
)
from .._alifestd_mark_num_leaves_asexual import (
    _alifestd_mark_num_leaves_asexual_fast_path,
)
from .._alifestd_try_add_ancestor_id_col_polars import (
    alifestd_try_add_ancestor_id_col_polars,
)


def _tgs_cast_origin_time_float(origin_time: pl.Series) -> pl.Series:
    """Implementation detail for `tgs_build_exposure_df_polars`.

    Returns `origin_time` as float64, warning if any value does not round
    trip through float64, e.g., an integer above 2**53.
    """
    as_float = origin_time.cast(pl.Float64)
    if (
        not origin_time.dtype.is_float()
        and (
            as_float.cast(origin_time.dtype, strict=False) != origin_time
        ).any()
    ):
        warnings.warn(
            "origin_time values lose precision as float64, so episode "
            "lengths may be inexact",
        )
    return as_float


def _tgs_calc_trait_states(
    *,
    ancestor_ids: np.ndarray,
    trait_transitions: np.ndarray,
    trait_name: str,
) -> np.ndarray:
    """Implementation detail for `tgs_build_exposure_df_polars`.

    Returns each node's trait state (0 or 1), summing its lineage's
    transitions from the root down.
    """
    is_root = ancestor_ids == np.arange(len(ancestor_ids))
    if (
        not np.isin(trait_transitions, (0, 1)).all()
        or trait_transitions[is_root].any()
    ):
        raise NotImplementedError(
            f"trait transitions {trait_name} must be 0 or +1, without nulls, "
            "and 0 at roots; losses and traits present at a root are not "
            "supported",
        )
    states = _alifestd_mark_lineage_cumsum_asexual_fast_path(
        ancestor_ids, trait_transitions.astype(np.int64), reverse=False
    )
    if states.max(initial=0) > 1:
        raise ValueError(
            f"trait transitions {trait_name} repeat gains along a lineage",
        )
    return states


def tgs_build_exposure_df_polars(
    phylogeny_df: pl.DataFrame,
    *,
    trait_transitions: typing.Sequence[str],
    episode_mask: typing.Optional[str],
    calibration_partition_by: typing.Optional[str],
    calibration_compare_by_bins: typing.Optional[str],
    calibration_compare_by_vals: typing.Sequence[str],
) -> pl.DataFrame:
    """Implementation detail for `alifestd_tgs_estimate_polars`.

    Returns one row per exposure episode, i.e., per non-root node's branch
    where `episode_mask`, if given, is true. See
    `alifestd_tgs_estimate_polars` for parameters and input schema.

    Output Schema
    -------------
    'origin_time' : float
        Origin time of the episode's end node.
    'episode_length' : float
        Duration elapsed between the origin times of the episode's start
        node (its end node's parent) and its end node.
    'survival_duration' : float
        Time from the end node to the latest leaf below it (0 for a leaf).
    'clade_size' : integer
        Number of leaves below the end node (1 for a leaf).
    'trait{i}_starts_absent' : boolean, per trait `i`
        Whether the episode started without the trait.
    'trait{i}_gain' : boolean, per trait `i`
        Whether the episode gained the trait.
    'partition_id' : integer
        Calibration partition, numbered from 0.
    'bin_id' : integer
        Calibration bin, numbered from 0, partition-major.
    'calibration_compare_by_{name}' : per `calibration_compare_by_vals` name
        Episode column `name`, if any, with `branch_length` naming
        `episode_length`; else the end node's `phylogeny_df` value.
    """
    phylogeny_df = alifestd_try_add_ancestor_id_col_polars(phylogeny_df)
    if "ancestor_id" not in phylogeny_df.lazy().collect_schema().names():
        raise NotImplementedError("only asexual phylogenies are supported")

    computed_cols = (
        "origin_time", "episode_length", "survival_duration", "clade_size",
    )  # fmt: skip
    calibration_cols = (
        calibration_partition_by,
        calibration_compare_by_bins,
        *calibration_compare_by_vals,
    )
    # affinity RE https://github.com/pola-rs/polars/issues/30408
    # TODO: drop the engine affinity for polars>2.1.0
    with pl.Config(engine_affinity="in-memory"):
        nodes_df = (
            phylogeny_df.lazy()
            .select(
                cs.by_name("id", "ancestor_id", "origin_time")
                | cs.by_name(*trait_transitions)
                | cs.by_name(
                    {episode_mask, *calibration_cols}
                    - {None, "branch_length", *computed_cols}
                ),
            )
            .collect()
        )

    log_memory_usage(logging.info)

    if not alifestd_has_contiguous_ids_polars(nodes_df):
        raise NotImplementedError("non-contiguous ids not supported")
    if not alifestd_is_topologically_sorted_polars(nodes_df):
        raise NotImplementedError(
            "non-topologically-sorted data not supported",
        )
    # a root may have one child, as each branch spans a node and its parent
    if alifestd_count_unifurcations_polars(
        nodes_df
    ) > alifestd_count_unifurcating_roots_polars(nodes_df):
        raise NotImplementedError("unifurcations not supported")

    ancestor_ids = nodes_df["ancestor_id"].to_numpy().astype(np.int64)
    origin_time = _tgs_cast_origin_time_float(
        nodes_df["origin_time"]
    ).to_numpy()  # nulls become nan
    if not np.isfinite(origin_time).all():
        raise NotImplementedError(
            "null, nan, and infinite origin_time values not supported",
        )
    # latest leaf below each node, or the node itself
    latest_ot = _alifestd_mark_max_descendant_origin_time_asexual_fast_path(
        ancestor_ids, origin_time
    )
    node_cols = {
        "origin_time": origin_time,
        "episode_length": origin_time - origin_time[ancestor_ids],
        "survival_duration": latest_ot - origin_time,
        "clade_size": _alifestd_mark_num_leaves_asexual_fast_path(
            ancestor_ids
        ),
    }
    for i, trait_name in enumerate(trait_transitions):
        transitions = nodes_df[trait_name].to_numpy()
        states = _tgs_calc_trait_states(
            ancestor_ids=ancestor_ids,
            trait_transitions=transitions,
            trait_name=trait_name,
        )
        node_cols[f"trait{i}_starts_absent"] = states[ancestor_ids] == 0
        node_cols[f"trait{i}_gain"] = transitions == 1

    # a root has no parent, so no branch over which to gain a trait
    is_episode = ancestor_ids != np.arange(len(ancestor_ids))
    if episode_mask is not None:
        mask = nodes_df[episode_mask]
        if mask.dtype != pl.Boolean or mask.has_nulls():
            raise NotImplementedError(
                "episode_mask column must be boolean, without nulls",
            )
        is_episode &= mask.to_numpy()

    # episode columns take precedence over phylogeny_df's
    episode_cols = [*node_cols]
    episodes_df = nodes_df.with_columns(pl.DataFrame(node_cols)).filter(
        is_episode
    )
    del nodes_df, node_cols  # clear memory
    gc.collect()
    log_memory_usage(logging.info)
    if episodes_df.is_empty():  # e.g., all masked out
        raise ValueError("phylogeny_df has no exposure episodes to fit")
    if sum(episodes_df.null_count().row(0)):
        raise NotImplementedError("null values not supported")
    if not (episodes_df["episode_length"] > 0).all():
        raise NotImplementedError(
            "exposure episodes with zero or negative lengths not supported; "
            "exclude them with episode_mask",
        )

    partition, bin_ = (
        pl.lit(0) if col is None else pl.col(col)
        for col in (calibration_partition_by, calibration_compare_by_bins)
    )
    partition_bin = pl.struct(partition.alias("partition"), bin_.alias("bin"))
    exposure_df = episodes_df.select(
        *episode_cols,
        (partition.rank("dense") - 1).cast(pl.Int64).alias("partition_id"),
        (partition_bin.rank("dense") - 1).cast(pl.Int64).alias("bin_id"),
        *(
            # branch_length names the episode's own length
            pl.col({"branch_length": "episode_length"}.get(col, col)).alias(
                f"calibration_compare_by_{col}"
            )
            for col in calibration_compare_by_vals
        ),
    )
    del episodes_df  # clear memory
    gc.collect()
    log_memory_usage(logging.info)

    times = exposure_df.select(
        "origin_time", "episode_length", "survival_duration"
    )
    assert np.isfinite(times.to_numpy()).all()
    assert (times["survival_duration"] >= 0).all()
    return exposure_df
