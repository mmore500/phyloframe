import typing

import numpy as np
import pandas as pd

from .._alifestd_collapse_unifurcations import alifestd_collapse_unifurcations
from .._alifestd_mark_origin_time_delta_asexual import (
    alifestd_mark_origin_time_delta_asexual,
)
from .._alifestd_to_working_format import alifestd_to_working_format


def tgs_prepare_asexual(
    phylogeny_df: pd.DataFrame,
    mutate: bool,
    *,
    trait_masks: typing.Sequence[np.ndarray],
    trait_names: typing.Sequence[str],
) -> pd.DataFrame:
    """Implementation detail for
    `alifestd_tgs_pendantmdc_estimate_asexual` and
    `alifestd_tgs_allmdc_estimate_asexual`.

    Here, tgs stands for trait gain/selection.

    Validates trait masks, then bypasses unifurcations and converts
    `phylogeny_df` to working format. Returns the converted phylogeny with
    columns `ancestor_id`, `origin_time`, `__alifestd_tgs_branch_length`
    (origin time delta from each node's parent), and one boolean column
    `__alifestd_tgs_trait{i}` per trait mask, realigned to its rows.
    """
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
    # carry masks as columns, as unifurcation collapse and working format
    # conversion drop and reorder rows
    for i, mask in enumerate(trait_masks):
        phylogeny_df[f"__alifestd_tgs_trait{i}"] = mask
    phylogeny_df = alifestd_collapse_unifurcations(
        phylogeny_df, mutate=True, ignore_topological_sensitivity=True
    )
    phylogeny_df = alifestd_to_working_format(phylogeny_df, mutate=True)
    return alifestd_mark_origin_time_delta_asexual(
        phylogeny_df, mutate=True, mark_as="__alifestd_tgs_branch_length"
    )
