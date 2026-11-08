import gc
import logging
import typing

import numpy as np
import polars as pl

from ..._auxlib._log_memory_usage import log_memory_usage
from .._alifestd_assign_contiguous_ids_polars import (
    alifestd_assign_contiguous_ids_polars,
)
from .._alifestd_collapse_unifurcations_polars import (
    alifestd_collapse_unifurcations_polars,
)
from .._alifestd_has_contiguous_ids_polars import (
    alifestd_has_contiguous_ids_polars,
)
from .._alifestd_is_topologically_sorted_polars import (
    alifestd_is_topologically_sorted_polars,
)
from .._alifestd_try_add_ancestor_id_col_polars import (
    alifestd_try_add_ancestor_id_col_polars,
)
from ._tgs_build_exposure_df import tgs_build_exposure_df
from ._tgs_calc_trait_states import tgs_calc_trait_states

# columns of `tgs_build_exposure_df` usable as `calibration_compare_by_vals`
_EXPOSURE_COLUMNS = frozenset(
    ("origin_time", "branch_length", "survival_duration", "clade_size"),
)


def _tgs_select_columns(
    phylogeny_df: pl.DataFrame,
    *,
    columns: typing.Sequence[str],
) -> pl.DataFrame:
    """Implementation detail for `tgs_build_exposure_df_polars`.

    Returns `phylogeny_df`'s `id`, `ancestor_id`, `origin_time`, and
    `columns`, collected, checking that ids are contiguous and topologically
    sorted.
    """
    selected = (
        phylogeny_df.lazy()
        .select("id", "ancestor_id", "origin_time", *columns)
        .collect()
    )
    if not selected.is_empty():
        if not alifestd_has_contiguous_ids_polars(selected):
            raise NotImplementedError("non-contiguous ids not supported")
        if not alifestd_is_topologically_sorted_polars(selected):
            raise NotImplementedError(
                "non-topologically-sorted data not supported",
            )
    return selected


def _tgs_calc_states(
    selected: pl.DataFrame, *, trait_transitions: typing.Sequence[str]
) -> typing.List[np.ndarray]:
    """Implementation detail for `tgs_build_exposure_df_polars`.

    Returns each trait's state at every node, summing its transitions (see
    `tgs_calc_trait_states`).
    """
    ancestor_ids = selected["ancestor_id"].to_numpy()
    states = []
    for column in trait_transitions:
        if selected[column].has_nulls():
            raise NotImplementedError(
                f"null values in trait transitions {column} not supported",
            )
        states.append(
            tgs_calc_trait_states(
                ancestor_ids=ancestor_ids,
                trait_transitions=selected[column].to_numpy(),
                trait_name=column,
            ),
        )
    return states


def _tgs_collapse_unifurcations(
    phylogeny_df: pl.DataFrame,
    *,
    states: typing.Sequence[np.ndarray],
    columns: typing.Sequence[str],
) -> pl.DataFrame:
    """Implementation detail for `tgs_build_exposure_df_polars`.

    Returns `selected`'s `columns` with unifurcations collapsed and ids made
    contiguous, carrying each node's original id, `__alifestd_tgs_node_id`,
    and trait states, `trait{i}_state`.
    """
    return (
        phylogeny_df.select("id", "ancestor_id", "origin_time", *columns)
        .with_columns(
            pl.col("id").alias("__alifestd_tgs_node_id"),
            *(pl.Series(f"trait{i}_state", s) for i, s in enumerate(states)),
        )
        .pipe(
            alifestd_collapse_unifurcations_polars,
            ignore_topological_sensitivity=True,
        )
        .pipe(alifestd_assign_contiguous_ids_polars)
    )


def _tgs_get_integer_column(values: pl.Series, *, column: str) -> pl.Series:
    """Implementation detail for `tgs_build_exposure_df_polars`.

    Returns calibration column `values`, checking they are integers without
    nulls.
    """
    if not values.dtype.is_integer() or values.has_nulls():
        raise NotImplementedError(
            f"calibration column {column} must be integer, without null "
            "values",
        )
    return values


def tgs_build_exposure_df_polars(
    phylogeny_df: pl.DataFrame,
    *,
    trait_transitions: typing.Sequence[str],
    caller: str,
    calibration_partition_by: typing.Optional[str] = None,
    calibration_compare_by_bins: typing.Optional[str] = None,
    calibration_compare_by_vals: typing.Sequence[str] = (),
) -> pl.DataFrame:
    """Implementation detail for `alifestd_tgs_pendantmdc_estimate_polars` and
    `alifestd_tgs_allmdc_estimate_polars`. Acronym tgs stands for trait
    gain/selection.

    Sums each trait's transitions into node states (see
    `tgs_calc_trait_states`), collapses unifurcations, and returns the
    phylogeny's exposure table, one row per exposure episode candidate,
    i.e., per non-root node's branch; see Output Schema. `caller` names the
    calling function in error messages.

    Input Schema
    ------------
    'id' : integer
        Taxon identifier; contiguous and topologically sorted.
    'ancestor_id' : integer
        Parent's taxon identifier (or own, for a root).
    'origin_time' : float
        Time of each taxon's origin.
    `trait_transitions` columns : integer
        Each node's trait state transition from its parent.
    `calibration_partition_by`, `calibration_compare_by_bins` columns : integer, optional
        Each node's calibration partition and bin.
    `calibration_compare_by_vals` columns : numeric, optional
        Values by which calibration bins are compared, unless exposure
        table columns (see below).

    Output Schema
    -------------
    'origin_time' : float
        Origin time of the episode's end node.
    'branch_length' : float
        Episode length.
    'survival_duration' : float
        Time from the end node to the latest leaf below it (0 for a leaf).
    'clade_size' : integer
        Number of leaves below the end node (1 for a leaf).
    'trait{i}_starts_absent' : boolean, per trait `i`
        Whether the episode started without the trait.
    'trait{i}_gain' : boolean, per trait `i`
        Whether the episode gained the trait.
    'node_id' : integer
        Id of the episode's end node in `phylogeny_df`.
    'calibration_partition' : integer, optional
        The end node's `calibration_partition_by` value, if given.
    'calibration_bin' : integer, optional
        The end node's `calibration_compare_by_bins` value, if given.
    'calibration_compare_by_{name}' : numeric, per `calibration_compare_by_vals` name
        The episode's value of that column, from the exposure table's own
        columns (e.g., `branch_length`, `clade_size`) if there, else from
        `phylogeny_df`'s, at the end node.
    """
    extra_columns = sorted(
        {
            calibration_partition_by,
            calibration_compare_by_bins,
            *calibration_compare_by_vals,
        }
        - _EXPOSURE_COLUMNS
        - {None},
    )
    # affinity RE https://github.com/pola-rs/polars/issues/30408
    # TODO: drop the engine affinity for polars>2.1.0
    with pl.Config(engine_affinity="in-memory"):
        phylogeny_df = alifestd_try_add_ancestor_id_col_polars(phylogeny_df)
        if "ancestor_id" not in phylogeny_df.lazy().collect_schema().names():
            raise NotImplementedError(
                f"{caller} only supports asexual phylogenies.",
            )
        selected = _tgs_select_columns(
            phylogeny_df, columns=[*trait_transitions, *extra_columns]
        )
        if selected.is_empty():
            return tgs_build_exposure_df(
                pl.DataFrame(
                    schema={
                        "ancestor_id": pl.Int64,
                        "origin_time": pl.Float64,
                        **{
                            f"trait{i}_state": pl.Int64
                            for i in range(len(trait_transitions))
                        },
                    },
                ),
            )
        states = _tgs_calc_states(
            selected, trait_transitions=trait_transitions
        )
        collapsed = _tgs_collapse_unifurcations(
            selected, states=states, columns=extra_columns
        )
        del selected, states
        gc.collect()
        log_memory_usage(logging.info)

    exposure_df = tgs_build_exposure_df(
        collapsed.select(
            "ancestor_id",
            "origin_time",
            *(f"trait{i}_state" for i in range(len(trait_transitions))),
        ),
    )
    # without roots, rows match exposure_df's
    collapsed = collapsed.filter(pl.col("ancestor_id") != pl.col("id"))
    columns = {"node_id": collapsed["__alifestd_tgs_node_id"]}
    if calibration_partition_by is not None:
        columns["calibration_partition"] = _tgs_get_integer_column(
            collapsed[calibration_partition_by],
            column=calibration_partition_by,
        )
    if calibration_compare_by_bins is not None:
        columns["calibration_bin"] = _tgs_get_integer_column(
            collapsed[calibration_compare_by_bins],
            column=calibration_compare_by_bins,
        )
    for column in calibration_compare_by_vals:
        if column in _EXPOSURE_COLUMNS:
            values = exposure_df[column]
        else:
            values = collapsed[column]
            if not values.dtype.is_numeric() or values.has_nulls():
                raise NotImplementedError(
                    f"calibration column {column} must be numeric, without "
                    "null values",
                )
        columns[f"calibration_compare_by_{column}"] = values
    exposure_df = exposure_df.with_columns(**columns)
    del collapsed, columns
    gc.collect()
    log_memory_usage(logging.info)
    return exposure_df
