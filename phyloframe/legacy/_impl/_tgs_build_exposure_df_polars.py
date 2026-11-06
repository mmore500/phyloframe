import typing

import numpy as np
import pandas as pd
import polars as pl

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
from ._tgs_calc_exposure_df import tgs_calc_exposure_df
from ._tgs_calc_trait_states import tgs_calc_trait_states


def tgs_build_exposure_df_polars(
    phylogeny_df: pl.DataFrame,
    *,
    trait_transitions: typing.Sequence[str],
    caller: str,
) -> pd.DataFrame:
    """Implementation detail for `alifestd_tgs_pendantmdc_estimate_polars` and
    `alifestd_tgs_allmdc_estimate_polars`. Acronym tgs stands for trait
    gain/selection.

    Sums each trait's transitions into node states (see
    `tgs_calc_trait_states`), collapses unifurcations, and returns the
    phylogeny's branches (see `tgs_calc_exposure_df`). `caller` names the
    calling function in error messages.
    """
    phylogeny_df = alifestd_try_add_ancestor_id_col_polars(phylogeny_df)
    schema_names = phylogeny_df.lazy().collect_schema().names()
    if "ancestor_id" not in schema_names:
        raise NotImplementedError(
            f"{caller} only supports asexual phylogenies.",
        )
    for column in trait_transitions:
        if column not in schema_names:
            raise ValueError(f"phylogeny_df has no column {column}")

    # the streaming engine panics on some of these plans; see
    # https://github.com/pola-rs/polars/issues/30408
    # TODO: drop the engine affinity once a polars release includes the fix
    # (after 2.1.0)
    with pl.Config(engine_affinity="in-memory"):
        selected = (
            phylogeny_df.lazy()
            .select("id", "ancestor_id", "origin_time", *trait_transitions)
            .collect()
        )
        if selected.is_empty():
            return tgs_calc_exposure_df(
                ancestor_ids=np.zeros(0, dtype=np.int64),
                origin_time=np.zeros(0),
                trait_states=[np.zeros(0, dtype=np.int64)]
                * len(trait_transitions),
            )
        if not alifestd_has_contiguous_ids_polars(selected):
            raise NotImplementedError("non-contiguous ids not supported")
        if not alifestd_is_topologically_sorted_polars(selected):
            raise NotImplementedError(
                "non-topologically-sorted data not supported",
            )
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
                    transitions=selected[column].to_numpy(),
                    trait_name=column,
                ),
            )
        collapsed = (
            alifestd_assign_contiguous_ids_polars(
                alifestd_collapse_unifurcations_polars(
                    selected.select(
                        "id", "ancestor_id", "origin_time"
                    ).with_columns(
                        pl.Series(f"__alifestd_tgs_state{i}", s)
                        for i, s in enumerate(states)
                    ),
                    ignore_topological_sensitivity=True,
                ),
            )
            .lazy()
            .collect()
        )
    return tgs_calc_exposure_df(
        ancestor_ids=collapsed["ancestor_id"].to_numpy(),
        origin_time=collapsed["origin_time"].to_numpy(),
        trait_states=[
            collapsed[f"__alifestd_tgs_state{i}"].to_numpy()
            for i in range(len(trait_transitions))
        ],
    )
