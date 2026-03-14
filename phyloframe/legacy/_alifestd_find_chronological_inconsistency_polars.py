import typing

import polars as pl

from ._alifestd_has_contiguous_ids_polars import (
    alifestd_has_contiguous_ids_polars,
)
from ._alifestd_try_add_ancestor_id_col_polars import (
    alifestd_try_add_ancestor_id_col_polars,
)


def alifestd_find_chronological_inconsistency_polars(
    phylogeny_df: pl.DataFrame,
) -> typing.Optional[int]:
    """Return the id of a taxon with origin time preceding its parent's, if
    any are present.
    """

    phylogeny_df = alifestd_try_add_ancestor_id_col_polars(phylogeny_df)

    df = phylogeny_df.lazy().collect()

    if df.is_empty():
        return None

    schema_names = df.columns
    if "ancestor_id" not in schema_names or "origin_time" not in schema_names:
        return None

    if alifestd_has_contiguous_ids_polars(phylogeny_df):
        ancestor_ids = df["ancestor_id"].to_numpy()
        origin_times = df["origin_time"].to_numpy()

        for id_, ancestor_id in enumerate(ancestor_ids):
            if origin_times[ancestor_id] > origin_times[id_]:
                return int(df["id"][id_])
        return None
    else:
        ids = df["id"].to_numpy()
        ancestor_ids = df["ancestor_id"].to_numpy()
        origin_times = df["origin_time"].to_numpy()

        origin_time_lookup = {}
        for id_, origin_time in zip(ids, origin_times):
            origin_time_lookup[int(id_)] = origin_time

        for id_, ancestor_id in zip(ids, ancestor_ids):
            if (
                origin_time_lookup[int(ancestor_id)]
                > origin_time_lookup[int(id_)]
            ):
                return int(id_)
        return None
