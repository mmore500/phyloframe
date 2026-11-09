import gc
import logging
import typing

import numpy as np
import polars as pl
import polars.selectors as cs

from ..._auxlib._log_memory_usage import log_memory_usage
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
from .._alifestd_mark_num_children_asexual import (
    _alifestd_mark_num_children_asexual_fast_path,
)
from .._alifestd_mark_num_leaves_asexual import (
    _alifestd_mark_num_leaves_asexual_fast_path,
)
from .._alifestd_try_add_ancestor_id_col_polars import (
    alifestd_try_add_ancestor_id_col_polars,
)


def _tgs_build_exposure_df(
    phylogeny_df: pl.DataFrame, *, is_kept_episode: np.ndarray
) -> pl.DataFrame:
    """Implementation detail for `tgs_build_exposure_df_polars`. Acronym tgs
    stands for trait gain/selection.

    Returns the exposure table of `phylogeny_df`, a phylogeny with
    contiguous, topologically sorted ids and no unifurcations, with one row
    per branch, i.e., per non-root node, where per-node boolean
    `is_kept_episode` is true; see Input Schema and Output
    Schema.

    Clade sizes and survival durations count the whole tree, masked-out
    nodes included; nodes with unknown (nan) origin times are skipped in
    finding each node's latest descendant.

    Raises ValueError if a kept exposure episode's branch length is not
    positive and finite, or its survival duration is not finite;
    masked-out episodes are not checked.

    Input Schema
    ------------
    'ancestor_id' : integer
        Parent's id (or own, for a root).
    'origin_time' : float
        Time of each node's origin.
    '__alifestd_tgs_trait{i}_state' : integer, per trait `i`
        Whether the node has trait `i` (0 or 1).

    Output Schema
    -------------
    'origin_time' : float
        Origin time of the branch's end node.
    'branch_length' : float
        Branch length.
    'survival_duration' : float
        Time from the end node to the latest leaf below it with known
        origin time (0 for a leaf), for which a trait gained on the branch
        must persist to be seen at every leaf of its clade.
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
    keep = (ancestor_ids != np.arange(len(ancestor_ids))) & is_kept_episode
    is_bad = ~(np.isfinite(branch_length) & (branch_length > 0))
    (bad_ids,) = np.nonzero(keep & is_bad)
    if len(bad_ids):
        raise ValueError(
            "branch lengths must be positive and finite, but exposure "
            f"episodes ending at nodes (e.g., ids {bad_ids[:5].tolist()}) "
            "have zero, negative, infinite, or undefined (nan) branch "
            "length; a parent with unknown (nan) origin_time, such as a "
            "root, leaves its children's branch lengths undefined; exclude "
            "such episodes with episode_mask",
        )
    # latest descendant, or self, with known origin time
    latest_ot = _alifestd_mark_max_descendant_origin_time_asexual_fast_path(
        ancestor_ids, np.where(np.isnan(origin_time), -np.inf, origin_time)
    )
    survival_duration = latest_ot - origin_time
    (bad_ids,) = np.nonzero(keep & ~np.isfinite(survival_duration))
    if len(bad_ids):
        raise ValueError(
            "survival durations must be finite, but exposure episodes ending "
            f"at nodes (e.g., ids {bad_ids[:5].tolist()}) have infinite or "
            "undefined (nan) survival duration, e.g., from an infinite "
            "descendant origin_time",
        )
    columns = {
        "origin_time": origin_time[keep],
        "branch_length": branch_length[keep],
        "survival_duration": survival_duration[keep],
        "clade_size": _alifestd_mark_num_leaves_asexual_fast_path(
            ancestor_ids
        )[keep],
    }
    num_traits = phylogeny_df.select(
        cs.matches(r"^__alifestd_tgs_trait\d+_state$")
    ).width
    for i in range(num_traits):
        states = phylogeny_df[f"__alifestd_tgs_trait{i}_state"].to_numpy()
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
    """Implementation detail for `_tgs_calc_states`. Acronym tgs stands for
    trait gain/selection.

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


def _tgs_validate_no_nulls(
    df: pl.DataFrame, *, cols: typing.Sequence[str], description: str
) -> None:
    """Implementation detail for `tgs_build_exposure_df_polars`.

    Raises NotImplementedError if any of `df`'s `cols`, described by
    `description` in the error message, has null values.
    """
    cols_with_nulls = (
        df.select(pl.col([*dict.fromkeys(cols)]).has_nulls())
        .unpivot()
        .filter("value")["variable"]
        .to_list()
    )
    if cols_with_nulls:
        raise NotImplementedError(
            f"null values in {description} {cols_with_nulls} not supported",
        )


def _tgs_validate_required_cols(
    schema_names: typing.Sequence[str],
    *,
    required_cols: typing.Dict[typing.Optional[str], str],
) -> None:
    """Implementation detail for `tgs_build_exposure_df_polars`.

    Raises ColumnNotFoundError if any of `required_cols`, column names
    mapped to the arguments that name them, are not among `schema_names`;
    a None column name is not required.
    """
    required_cols = {
        col: argument
        for col, argument in required_cols.items()
        if col is not None
    }
    missing = set(required_cols) - set(schema_names)
    if missing:
        raise pl.exceptions.ColumnNotFoundError(
            "missing required columns: "
            + ", ".join(
                f"{col!r} ({required_cols[col]})" for col in sorted(missing)
            ),
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
    _tgs_validate_no_nulls(
        selected, cols=trait_transition_cols, description="trait transitions"
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


def _tgs_check_no_unifurcations(selected: pl.DataFrame) -> None:
    """Implementation detail for `tgs_build_exposure_df_polars`.

    Raises NotImplementedError if a non-root node has exactly one child, so
    that each branch spans from a node to its parent; a root may have one
    child.
    """
    ancestor_ids = selected["ancestor_id"].to_numpy()
    is_root = ancestor_ids == np.arange(len(ancestor_ids))
    num_children = _alifestd_mark_num_children_asexual_fast_path(ancestor_ids)
    (unifurcation_ids,) = np.nonzero((num_children == 1) & ~is_root)
    if len(unifurcation_ids):
        raise NotImplementedError(
            "unifurcations, i.e., non-root nodes with one child (e.g., ids "
            f"{unifurcation_ids[:5].tolist()}), are not yet supported",
        )


def _tgs_add_trait_state_cols(
    selected: pl.DataFrame, *, trait_transition_cols: typing.Sequence[str]
) -> pl.DataFrame:
    """Implementation detail for `tgs_build_exposure_df_polars`.

    Returns `selected` with trait transitions replaced by trait states,
    `__alifestd_tgs_trait{i}_state` (see `_tgs_calc_states`).
    """
    states = _tgs_calc_states(
        selected, trait_transition_cols=trait_transition_cols
    )
    return selected.drop(trait_transition_cols).with_columns(
        pl.Series(f"__alifestd_tgs_trait{i}_state", s)
        for i, s in enumerate(states)
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
    }
    if invalid_cols:
        raise NotImplementedError(
            f"calibration columns {sorted(invalid_cols)} must be "
            "numeric (integer, for partitions and bins)",
        )
    _tgs_validate_no_nulls(
        exposure_df,
        cols=[*integer_cols, *numeric_cols],
        description="calibration columns",
    )


def _tgs_calc_is_kept_episode(
    nodes_df: pl.DataFrame, *, episode_mask: typing.Optional[str]
) -> np.ndarray:
    """Implementation detail for `tgs_build_exposure_df_polars`.

    Returns, per node, whether to fit the exposure episode ending there: the
    boolean `episode_mask` column, if given, else all true. Values at roots,
    which end no episode, are ignored.
    """
    if episode_mask is None:
        return np.ones(len(nodes_df), dtype=bool)
    mask = nodes_df[episode_mask]
    is_root = nodes_df["id"] == nodes_df["ancestor_id"]
    if mask.dtype != pl.Boolean or mask.filter(~is_root).has_nulls():
        raise NotImplementedError(
            f"episode mask column {episode_mask} must be boolean, without "
            "null values",
        )
    return mask.fill_null(False).to_numpy()


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
    `_tgs_calc_trait_states`) and returns the phylogeny's exposure table, one
    row per exposure episode candidate, i.e., per non-root node's branch, kept
    if `episode_mask`, if given, is true at its end node; see Output Schema.
    Branch lengths and calibration columns are checked only for kept episodes.
    `caller` names the calling function in error messages.

    Raises ValueError if no exposure episodes are kept, e.g., if
    `phylogeny_df` is empty or all are masked out, and NotImplementedError
    if it has unifurcations other than roots with one child.

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
        Time from the end node to the latest leaf below it with known
        origin time (0 for a leaf).
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
        schema_names = phylogeny_df.lazy().collect_schema().names()
        if "ancestor_id" not in schema_names:
            raise NotImplementedError(
                f"{caller} only supports asexual phylogenies.",
            )
        _tgs_validate_required_cols(
            schema_names,
            required_cols={
                **{col: "trait_transitions" for col in trait_transition_cols},
                episode_mask: "episode_mask",
                calibration_partition_by: "calibration_partition_by",
                calibration_compare_by_bins: "calibration_compare_by_bins",
            },
        )
        selected = _tgs_select_columns(
            phylogeny_df,
            trait_transition_cols=trait_transition_cols,
            calibration_cols=[
                *calibration_cols.values(),
                *([episode_mask] if episode_mask is not None else []),
            ],
        )
    _tgs_check_no_unifurcations(selected)
    node_states_df = _tgs_add_trait_state_cols(
        selected, trait_transition_cols=trait_transition_cols
    )
    del selected
    gc.collect()
    log_memory_usage(logging.info)

    # masked-out episodes are dropped before their values are checked
    is_kept_episode = _tgs_calc_is_kept_episode(
        node_states_df, episode_mask=episode_mask
    )
    exposure_df = _tgs_build_exposure_df(
        node_states_df, is_kept_episode=is_kept_episode
    )
    # each kept episode's end-node values of calibration columns that
    # exposure_df lacks, row-aligned with exposure_df (roots end none)
    episode_node_cols_df = node_states_df.filter(
        (pl.col("ancestor_id") != pl.col("id")) & pl.Series(is_kept_episode)
    ).select(
        cs.by_name(*calibration_cols.values(), require_all=False)
        - cs.by_name(exposure_df.columns, require_all=False),
    )
    del node_states_df
    calibration_source_df = pl.concat(
        [exposure_df, episode_node_cols_df], how="horizontal"
    )
    del episode_node_cols_df
    _tgs_validate_calibration_cols(
        calibration_source_df,
        integer_cols=[
            col
            for col in (calibration_partition_by, calibration_compare_by_bins)
            if col is not None
        ],
        numeric_cols=calibration_compare_by_vals,
    )
    exposure_df = exposure_df.with_columns(
        calibration_source_df.select(
            pl.col(col).alias(name) for name, col in calibration_cols.items()
        ),
    )
    del calibration_source_df
    gc.collect()
    log_memory_usage(logging.info)
    if exposure_df.is_empty():  # e.g., all masked out
        raise ValueError("phylogeny_df has no exposure episodes to fit")
    return exposure_df
