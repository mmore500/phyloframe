import typing

import pandas as pd

from .._alifestd_collapse_unifurcations import alifestd_collapse_unifurcations
from .._alifestd_to_working_format import alifestd_to_working_format
from ._tgs_calc_branches import tgs_calc_branches
from ._tgs_calc_trait_states import tgs_calc_trait_states


def tgs_prepare_asexual(
    phylogeny_df: pd.DataFrame,
    mutate: bool,
    *,
    trait_transitions: typing.Sequence[str],
) -> pd.DataFrame:
    """Implementation detail for
    `alifestd_tgs_pendantmdc_estimate_asexual` and
    `alifestd_tgs_allmdc_estimate_asexual`.

    Here, tgs stands for trait gain/selection.

    Converts `phylogeny_df` to working format, sums each trait's
    transitions into node states (see `tgs_calc_trait_states`), bypasses
    unifurcations, and returns its branches (see `tgs_calc_branches`).
    Computing states before bypassing unifurcations credits a bypassed
    node's transition to the merged branch.
    """
    if not mutate:
        phylogeny_df = phylogeny_df.copy()

    for column in trait_transitions:
        if column not in phylogeny_df:
            raise ValueError(f"phylogeny_df has no column {column}")
    phylogeny_df = alifestd_to_working_format(phylogeny_df, mutate=True)
    for i, column in enumerate(trait_transitions):
        phylogeny_df[f"__alifestd_tgs_state{i}"] = tgs_calc_trait_states(
            ancestor_ids=phylogeny_df["ancestor_id"].to_numpy(),
            transitions=phylogeny_df[column].to_numpy(),
            trait_name=column,
        )
    phylogeny_df = alifestd_collapse_unifurcations(
        phylogeny_df, mutate=True, ignore_topological_sensitivity=True
    )
    phylogeny_df = alifestd_to_working_format(phylogeny_df, mutate=True)
    return tgs_calc_branches(
        ancestor_ids=phylogeny_df["ancestor_id"].to_numpy(),
        origin_time=phylogeny_df["origin_time"].to_numpy(),
        trait_states=[
            phylogeny_df[f"__alifestd_tgs_state{i}"].to_numpy()
            for i in range(len(trait_transitions))
        ],
    )
