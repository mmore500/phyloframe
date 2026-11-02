import typing

import numpy as np
import polars as pl

from .._alifestd_has_contiguous_ids_polars import (
    alifestd_has_contiguous_ids_polars,
)
from .._alifestd_is_topologically_sorted_polars import (
    alifestd_is_topologically_sorted_polars,
)
from .._alifestd_try_add_ancestor_id_col_polars import (
    alifestd_try_add_ancestor_id_col_polars,
)


def alifestd_mask_trait_polars_impl(
    phylogeny_df: pl.DataFrame,
    *,
    trait_mask: typing.Union[str, pl.Expr],
    mark_as: str,
    fast_path: typing.Callable[[np.ndarray, np.ndarray], np.ndarray],
    caller: str,
) -> pl.DataFrame:
    """Implementation detail for `alifestd_mask_trait_mdcs_strict_polars`
    and `alifestd_mask_trait_singletons_polars`.

    Validates `phylogeny_df` and `trait_mask`, then marks nodes by
    `fast_path(ancestor_ids, trait_mask)`.
    """
    if isinstance(trait_mask, str):
        trait_mask = pl.col(trait_mask)

    phylogeny_df = alifestd_try_add_ancestor_id_col_polars(phylogeny_df)
    schema_names = phylogeny_df.lazy().collect_schema().names()
    if "ancestor_id" not in schema_names:
        raise NotImplementedError(
            f"{caller} only supports asexual phylogenies.",
        )
    if phylogeny_df.lazy().limit(1).collect().is_empty():
        return phylogeny_df.with_columns(
            pl.lit(False, dtype=pl.Boolean).alias(mark_as),
        )
    if not alifestd_has_contiguous_ids_polars(phylogeny_df):
        raise NotImplementedError("non-contiguous ids not supported")
    if not alifestd_is_topologically_sorted_polars(phylogeny_df):
        raise NotImplementedError(
            "non-topologically-sorted data not supported",
        )

    selected = (
        phylogeny_df.lazy()
        .select(
            pl.col("ancestor_id"),
            trait_mask=trait_mask.cast(pl.Boolean),
        )
        .collect()
    )
    if selected["trait_mask"].has_nulls():
        raise NotImplementedError("null trait_mask values not supported")
    result = fast_path(
        selected["ancestor_id"].to_numpy(),
        selected["trait_mask"].to_numpy(),
    )
    return phylogeny_df.with_columns(pl.Series(name=mark_as, values=result))
