import typing

import numpy as np
import pandas as pd

from .._alifestd_collapse_unifurcations import alifestd_collapse_unifurcations
from .._alifestd_mark_origin_time_delta_asexual import (
    alifestd_mark_origin_time_delta_asexual,
)
from .._alifestd_to_working_format import alifestd_to_working_format


class TgsPrepared(typing.NamedTuple):
    """Implementation detail for
    `alifestd_estimate_trait_gain_selection_pendantmdc_asexual` and
    `alifestd_estimate_trait_gain_selection_allmdc_asexual`.

    Here, tgs stands for trait gain/selection.

    Per-node arrays of a phylogeny with contiguous, topologically sorted
    ids and no unifurcations, with each node's branch length (origin time
    delta to its parent), and trait masks aligned to its nodes.
    """

    ancestor_ids: np.ndarray
    origin_time: np.ndarray
    branch_lengths: np.ndarray
    trait_masks: typing.List[np.ndarray]
    trait_names: typing.List[str]


def tgs_prepare_asexual(
    phylogeny_df: pd.DataFrame,
    mutate: bool,
    *,
    trait_masks: typing.Sequence[np.ndarray],
    trait_names: typing.Optional[typing.Sequence[str]],
) -> TgsPrepared:
    """Implementation detail for
    `alifestd_estimate_trait_gain_selection_pendantmdc_asexual` and
    `alifestd_estimate_trait_gain_selection_allmdc_asexual`.

    Here, tgs stands for trait gain/selection.

    Validates trait masks and names, then bypasses unifurcations and
    converts `phylogeny_df` to working format, returning per-node arrays
    and trait masks realigned to the converted rows.
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
        phylogeny_df, mutate=True, mark_as="__alifestd_tgs_branch_length"
    )
    rows = phylogeny_df["__alifestd_tgs_row"].to_numpy()
    return TgsPrepared(
        ancestor_ids=phylogeny_df["ancestor_id"].to_numpy(),
        origin_time=phylogeny_df["origin_time"].to_numpy(),
        branch_lengths=phylogeny_df["__alifestd_tgs_branch_length"].to_numpy(),
        trait_masks=[mask[rows] for mask in trait_masks],
        trait_names=list(trait_names),
    )
