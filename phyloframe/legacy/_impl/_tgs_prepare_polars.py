import typing

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
from .._alifestd_mark_origin_time_delta_polars import (
    alifestd_mark_origin_time_delta_polars,
)
from .._alifestd_try_add_ancestor_id_col_polars import (
    alifestd_try_add_ancestor_id_col_polars,
)


def tgs_prepare_polars(
    phylogeny_df: pl.DataFrame,
    *,
    trait_masks: typing.Sequence[typing.Union[str, pl.Expr]],
    trait_names: typing.Sequence[str],
    caller: str,
) -> pl.DataFrame:
    """Implementation detail for
    `alifestd_tgs_pendantmdc_estimate_polars` and
    `alifestd_tgs_allmdc_estimate_polars`.

    Here, tgs stands for trait gain/selection.

    Evaluates trait masks and bypasses unifurcations. Returns the converted
    phylogeny with columns `ancestor_id`, `origin_time`,
    `__alifestd_tgs_branch_length` (origin time delta from each node's
    parent), and one boolean column `__alifestd_tgs_trait{i}` per trait
    mask. `caller` names the calling function in error messages.
    """
    trait_masks = [
        pl.col(mask) if isinstance(mask, str) else mask for mask in trait_masks
    ]

    phylogeny_df = alifestd_try_add_ancestor_id_col_polars(phylogeny_df)
    schema_names = phylogeny_df.lazy().collect_schema().names()
    if "ancestor_id" not in schema_names:
        raise NotImplementedError(
            f"{caller} only supports asexual phylogenies.",
        )

    # the streaming engine panics on some of these plans; see
    # https://github.com/pola-rs/polars/issues/30408
    # TODO: drop the engine affinity once a polars release includes the fix
    # (after 2.1.0)
    with pl.Config(engine_affinity="in-memory"):
        # evaluate masks before collapsing unifurcations renumbers rows
        selected = phylogeny_df.lazy().select(
            pl.col("id"),
            pl.col("ancestor_id"),
            pl.col("origin_time"),
            *(
                mask.cast(pl.Boolean).alias(f"__alifestd_tgs_trait{i}")
                for i, mask in enumerate(trait_masks)
            ),
        )
        if not selected.limit(1).collect().is_empty():
            if not alifestd_has_contiguous_ids_polars(selected):
                raise NotImplementedError(
                    "non-contiguous ids not supported",
                )
            if not alifestd_is_topologically_sorted_polars(selected):
                raise NotImplementedError(
                    "non-topologically-sorted data not supported",
                )
            selected = alifestd_assign_contiguous_ids_polars(
                alifestd_collapse_unifurcations_polars(
                    selected, ignore_topological_sensitivity=True
                ),
            )
        selected = (
            alifestd_mark_origin_time_delta_polars(
                selected, mark_as="__alifestd_tgs_branch_length"
            )
            .lazy()
            .collect()
        )

    for i, name in enumerate(trait_names):
        if selected[f"__alifestd_tgs_trait{i}"].has_nulls():
            raise NotImplementedError(
                f"null values in trait mask {name} not supported",
            )
    return selected
