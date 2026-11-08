import gc
import logging
import typing

import numpy as np
import polars as pl
import polars.selectors as cs

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
from .._alifestd_mark_lineage_cumsum_asexual import (
    _alifestd_mark_lineage_cumsum_asexual_fast_path,
)
from .._alifestd_mark_lineage_delta_asexual import (
    _alifestd_mark_lineage_delta_asexual_fast_path,
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


def _tgs_build_exposure_df(phylogeny_df: pl.DataFrame) -> pl.DataFrame:
    """Implementation detail for `alifestd_tgs_estimate_polars`. Acronym tgs
    stands for trait gain/selection.

    Returns the exposure table of `phylogeny_df`, a phylogeny with
    contiguous, topologically sorted ids and no unifurcations, with one row
    per branch, i.e., per non-root node; see Input Schema and Output
    Schema.

    Raises ValueError if a non-root node's branch has zero, negative, or
    undefined (nan) length.

    Input Schema
    ------------
    'ancestor_id' : integer
        Parent's id (or own, for a root).
    'origin_time' : float
        Time of each node's origin.
    'trait{i}_state' : integer, per trait `i`
        Whether the node has trait `i` (0 or 1).

    Output Schema
    -------------
    'origin_time' : float
        Origin time of the branch's end node.
    'branch_length' : float
        Branch length.
    'survival_duration' : float
        Time from the end node to the latest leaf below it (0 for a leaf),
        for which a trait gained on the branch must persist to be seen at
        every leaf of its clade.
    'clade_size' : integer
        Number of leaves below the end node (1 for a leaf).
    'trait{i}_starts_absent' : boolean, per trait `i`
        Whether the branch started without the trait (i.e., its parent
        lacks it), so could gain it on the branch.
    'trait{i}_gain' : boolean, per trait `i`
        Whether the branch gained the trait.
    """
    ancestor_ids = phylogeny_df["ancestor_id"].to_numpy().astype(np.int64)
    origin_time = phylogeny_df["origin_time"].to_numpy().astype(np.float64)
    branch_length = _alifestd_mark_lineage_delta_asexual_fast_path(
        ancestor_ids, origin_time
    )
    # a root has no parent, so no branch over which to gain a trait
    keep = ancestor_ids != np.arange(len(ancestor_ids))
    (bad_ids,) = np.nonzero(keep & ~(branch_length > 0))  # nan is bad
    if len(bad_ids):
        raise ValueError(
            "branch lengths must be positive, but non-root nodes (e.g., ids "
            f"{bad_ids[:5].tolist()}) have zero, negative, or undefined (nan) "
            "branch length; a parent with unknown (nan) origin_time, such "
            "as a root, leaves its children's branch lengths undefined",
        )
    latest_ot = _alifestd_mark_max_descendant_origin_time_asexual_fast_path(
        ancestor_ids, origin_time
    )
    columns = {
        "origin_time": origin_time[keep],
        "branch_length": branch_length[keep],
        "survival_duration": (latest_ot - origin_time)[keep],
        "clade_size": _alifestd_mark_num_leaves_asexual_fast_path(
            ancestor_ids
        )[keep],
    }
    num_traits = phylogeny_df.select(cs.matches(r"^trait\d+_state$")).width
    for i in range(num_traits):
        states = phylogeny_df[f"trait{i}_state"].to_numpy()
        deltas = _alifestd_mark_lineage_delta_asexual_fast_path(
            ancestor_ids, states
        )
        # a node's parent's state is its state minus its delta
        columns[f"trait{i}_starts_absent"] = (states - deltas == 0)[keep]
        columns[f"trait{i}_gain"] = (deltas == 1)[keep]
    return pl.DataFrame(columns)


def _tgs_calc_trait_states(
    *,
    ancestor_ids: np.ndarray,
    trait_name: str,
    trait_transitions: np.ndarray,
) -> np.ndarray:
    """Implementation detail for `alifestd_tgs_estimate_polars`. Acronym tgs
    stands for trait gain/selection.

    Takes a phylogeny with contiguous, topologically sorted ids and one
    trait's transitions (see `alifestd_mark_lineage_delta_asexual`).
    Returns each node's state, summing transitions from its root down, after
    checking that transitions are +1, -1, or 0, that none are losses (not
    yet supported), that roots lack the trait (zero root transitions), and
    that states stay 0 or 1.
    """
    trait_transitions = np.asarray(trait_transitions)
    if not (
        np.issubdtype(trait_transitions.dtype, np.integer)
        or trait_transitions.dtype == np.bool_
    ):
        raise NotImplementedError(
            f"trait transitions {trait_name} must have integer dtype, got "
            f"{trait_transitions.dtype}",
        )
    trait_transitions = trait_transitions.astype(np.int64)
    if not np.isin(trait_transitions, (-1, 0, 1)).all():
        raise ValueError(
            f"trait transitions {trait_name} must be +1, -1, or 0",
        )
    if (trait_transitions == -1).any():
        raise NotImplementedError(
            f"trait transitions {trait_name} include losses (-1), which are "
            "not yet supported",
        )
    ancestor_ids = np.asarray(ancestor_ids, dtype=np.int64)
    if trait_transitions[ancestor_ids == np.arange(len(ancestor_ids))].any():
        raise NotImplementedError(
            f"trait {trait_name} is present at a root (nonzero root "
            "transition); traits present at a root, so at every node absent "
            "loss, are uninformative, so drop them",
        )
    states = _alifestd_mark_lineage_cumsum_asexual_fast_path(
        ancestor_ids,
        trait_transitions,
        reverse=False,
    )
    if not np.isin(states, (0, 1)).all():
        raise ValueError(
            f"trait transitions {trait_name} give states other than 0 and 1, "
            "e.g., from duplicate gains along a lineage",
        )
    return states


def _tgs_get_cols_with_nulls(
    df: pl.DataFrame, *, cols: typing.Sequence[str]
) -> typing.List[str]:
    """Implementation detail for `tgs_build_exposure_df_polars`.

    Returns those of `df`'s `cols` with null values.
    """
    return (
        df.select(pl.col([*dict.fromkeys(cols)]).has_nulls())
        .unpivot()
        .filter("value")["variable"]
        .to_list()
    )


def _tgs_select_columns(
    phylogeny_df: pl.DataFrame,
    *,
    trait_transition_cols: typing.Sequence[str],
    calibration_cols: typing.Sequence[str],
) -> pl.DataFrame:
    """Implementation detail for `tgs_build_exposure_df_polars`.

    Returns `phylogeny_df`'s `id`, `ancestor_id`, `origin_time`, and
    `trait_transition_cols`, and those of `calibration_cols` it has,
    collected, checking that ids are contiguous and topologically sorted.
    """
    selected = (
        phylogeny_df.lazy()
        .select(
            cs.by_name(
                "id", "ancestor_id", "origin_time", *trait_transition_cols
            )
            # absent calibration columns may be exposure table columns
            | cs.by_name(*calibration_cols, require_all=False),
        )
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
    selected: pl.DataFrame, *, trait_transition_cols: typing.Sequence[str]
) -> typing.List[np.ndarray]:
    """Implementation detail for `tgs_build_exposure_df_polars`.

    Returns each trait's state at every node, summing its transitions (see
    `_tgs_calc_trait_states`).
    """
    cols_with_nulls = _tgs_get_cols_with_nulls(
        selected, cols=trait_transition_cols
    )
    if cols_with_nulls:
        raise NotImplementedError(
            f"null values in trait transitions {cols_with_nulls} not supported",
        )
    ancestor_ids = selected["ancestor_id"].to_numpy()
    return [
        _tgs_calc_trait_states(
            ancestor_ids=ancestor_ids,
            trait_transitions=selected[column].to_numpy(),
            trait_name=column,
        )
        for column in trait_transition_cols
    ]


def _tgs_collapse_unifurcations(
    selected: pl.DataFrame,
    *,
    trait_transition_cols: typing.Sequence[str],
    states: typing.Sequence[np.ndarray],
) -> pl.DataFrame:
    """Implementation detail for `tgs_build_exposure_df_polars`.

    Returns `selected`, with trait transitions replaced by trait states,
    `trait{i}_state`, and unifurcations collapsed and ids made contiguous.
    """
    return (
        selected.drop(trait_transition_cols)
        .with_columns(
            pl.Series(f"trait{i}_state", s) for i, s in enumerate(states)
        )
        .pipe(
            alifestd_collapse_unifurcations_polars,
            ignore_topological_sensitivity=True,
        )
        .pipe(alifestd_assign_contiguous_ids_polars)
    )


def _tgs_validate_calibration_cols(
    exposure_df: pl.DataFrame,
    *,
    integer_cols: typing.Sequence[str],
    numeric_cols: typing.Sequence[str],
) -> None:
    """Implementation detail for `tgs_build_exposure_df_polars`.

    Checks that `exposure_df`'s calibration columns are integer,
    `integer_cols`, or numeric, `numeric_cols`, without null values.
    """
    invalid_cols = {
        *exposure_df.select(pl.col(integer_cols))
        .select(~cs.integer())
        .columns,
        *exposure_df.select(pl.col(numeric_cols))
        .select(~cs.numeric())
        .columns,
        *_tgs_get_cols_with_nulls(
            exposure_df, cols=[*integer_cols, *numeric_cols]
        ),
    }
    if invalid_cols:
        raise NotImplementedError(
            f"calibration columns {sorted(invalid_cols)} must be "
            "numeric (integer, for partitions and bins), without null values",
        )


def _tgs_apply_episode_mask(
    exposure_df: pl.DataFrame, *, mask: pl.Series
) -> pl.DataFrame:
    """Implementation detail for `tgs_build_exposure_df_polars`.

    Returns the rows of `exposure_df` where boolean `mask`, the episode mask
    column at each episode's end node, is true.
    """
    if mask.dtype != pl.Boolean or mask.has_nulls():
        raise NotImplementedError(
            f"episode mask column {mask.name} must be boolean, without null "
            "values",
        )
    return exposure_df.filter(mask)


def tgs_build_exposure_df_polars(
    phylogeny_df: pl.DataFrame,
    *,
    trait_transition_cols: typing.Sequence[str],
    caller: str,
    calibration_partition_by: typing.Optional[str],
    calibration_compare_by_bins: typing.Optional[str],
    calibration_compare_by_vals: typing.Sequence[str],
    episode_mask: typing.Optional[str],
) -> pl.DataFrame:
    """Implementation detail for `alifestd_tgs_estimate_polars`. Acronym tgs
    stands for trait gain/selection.

    Sums each trait's transitions into node states (see
    `_tgs_calc_trait_states`), collapses unifurcations, and returns the
    phylogeny's exposure table, one row per exposure episode candidate,
    i.e., per non-root node's branch, kept if `episode_mask`, if given, is
    true at its end node; see Output Schema. `caller` names the calling
    function in error messages.

    Raises ValueError if `phylogeny_df` is empty.

    Input Schema
    ------------
    'id' : integer
        Taxon identifier; contiguous and topologically sorted.
    'ancestor_id' : integer
        Parent's taxon identifier (or own, for a root).
    'origin_time' : float
        Time of each taxon's origin.
    `trait_transition_cols` columns : integer
        Each node's trait state transition from its parent.
    `calibration_partition_by`, `calibration_compare_by_bins` columns : integer, optional
        Each node's calibration partition and bin.
    `calibration_compare_by_vals` columns : numeric, optional
        Values by which calibration bins are compared, unless exposure
        table columns (see below).
    `episode_mask` column : boolean, optional
        Whether to keep the exposure episode ending at each node.

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
    'calibration_partition' : integer, optional
        The end node's `calibration_partition_by` value, if given.
    'calibration_bin' : integer, optional
        The end node's `calibration_compare_by_bins` value, if given.
    'calibration_compare_by_{name}' : numeric, per `calibration_compare_by_vals` name
        The episode's value of that column, from the exposure table's own
        columns (e.g., `branch_length`, `clade_size`) if there, else from
        `phylogeny_df`'s, at the end node.
    """
    # output column name: source column name
    calibration_cols = {
        name: col
        for name, col in (
            ("calibration_partition", calibration_partition_by),
            ("calibration_bin", calibration_compare_by_bins),
            *(
                (f"calibration_compare_by_{col}", col)
                for col in calibration_compare_by_vals
            ),
        )
        if col is not None
    }
    # affinity RE https://github.com/pola-rs/polars/issues/30408
    # TODO: drop the engine affinity for polars>2.1.0
    with pl.Config(engine_affinity="in-memory"):
        phylogeny_df = alifestd_try_add_ancestor_id_col_polars(phylogeny_df)
        if "ancestor_id" not in phylogeny_df.lazy().collect_schema().names():
            raise NotImplementedError(
                f"{caller} only supports asexual phylogenies.",
            )
        selected = _tgs_select_columns(
            phylogeny_df,
            trait_transition_cols=trait_transition_cols,
            calibration_cols=[
                *calibration_cols.values(),
                *([episode_mask] if episode_mask is not None else []),
            ],
        )
        if selected.is_empty():
            raise ValueError("phylogeny_df has no exposure episodes to fit")
        collapsed = _tgs_collapse_unifurcations(
            selected,
            trait_transition_cols=trait_transition_cols,
            states=_tgs_calc_states(
                selected, trait_transition_cols=trait_transition_cols
            ),
        )
        del selected
        gc.collect()
        log_memory_usage(logging.info)

    exposure_df = _tgs_build_exposure_df(collapsed)
    # without roots, rows match exposure_df's; its own columns take
    # precedence
    node_df = collapsed.filter(pl.col("ancestor_id") != pl.col("id")).select(
        cs.by_name(
            *calibration_cols.values(),
            *([episode_mask] if episode_mask is not None else []),
            require_all=False,
        )
        - cs.by_name(exposure_df.columns, require_all=False),
    )
    del collapsed
    source_df = pl.concat([exposure_df, node_df], how="horizontal")
    del node_df
    _tgs_validate_calibration_cols(
        source_df,
        integer_cols=[
            col
            for col in (calibration_partition_by, calibration_compare_by_bins)
            if col is not None
        ],
        numeric_cols=calibration_compare_by_vals,
    )
    exposure_df = exposure_df.with_columns(
        source_df.select(
            pl.col(col).alias(name) for name, col in calibration_cols.items()
        ),
    )
    if episode_mask is not None:
        exposure_df = _tgs_apply_episode_mask(
            exposure_df, mask=source_df[episode_mask]
        )
    del source_df
    gc.collect()
    log_memory_usage(logging.info)
    return exposure_df
