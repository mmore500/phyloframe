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
from ._tgs_prepare_asexual import TgsPrepared


def tgs_prepare_polars(
    phylogeny_df: pl.DataFrame,
    *,
    trait_masks: typing.Sequence[typing.Union[str, pl.Expr]],
    trait_names: typing.Optional[typing.Sequence[str]],
    neutral_traits: typing.Sequence[typing.Union[int, str]],
    caller: str,
) -> typing.Tuple[TgsPrepared, typing.List[int]]:
    """Implementation detail for
    `alifestd_estimate_trait_gain_selection_pendantmdc_polars` and
    `alifestd_estimate_trait_gain_selection_allmdc_polars`.

    Here, tgs stands for trait gain/selection.

    Resolves trait names and neutral trait names to indices, evaluates
    trait masks, and bypasses unifurcations, returning per-node arrays and
    neutral trait indices. `caller` names the calling function in error
    messages.
    """
    if trait_names is None:
        trait_names = [
            mask if isinstance(mask, str) else str(i)
            for i, mask in enumerate(trait_masks)
        ]
    if len(trait_names) != len(trait_masks):
        raise ValueError(
            f"got {len(trait_names)} trait names for "
            f"{len(trait_masks)} trait masks",
        )
    neutral_traits = [
        list(trait_names).index(trait) if isinstance(trait, str) else trait
        for trait in neutral_traits
    ]

    trait_masks = [
        pl.col(mask) if isinstance(mask, str) else mask for mask in trait_masks
    ]

    phylogeny_df = alifestd_try_add_ancestor_id_col_polars(phylogeny_df)
    schema_names = phylogeny_df.lazy().collect_schema().names()
    if "ancestor_id" not in schema_names:
        raise NotImplementedError(
            f"{caller} only supports asexual phylogenies.",
        )

    # evaluate masks before collapsing unifurcations renumbers rows, then
    # continue eagerly to avoid https://github.com/pola-rs/polars/issues/30408
    # TODO: stay lazy once a polars release includes the fix (after 2.1.0)
    selected = (
        phylogeny_df.lazy()
        .select(
            pl.col("id"),
            pl.col("ancestor_id"),
            pl.col("origin_time"),
            *(
                mask.cast(pl.Boolean).alias(f"__alifestd_tgs_trait{i}")
                for i, mask in enumerate(trait_masks)
            ),
        )
        .collect()
    )
    if not selected.is_empty():
        if not alifestd_has_contiguous_ids_polars(selected):
            raise NotImplementedError(
                "non-contiguous ids not supported",
            )
        if not alifestd_is_topologically_sorted_polars(selected):
            raise NotImplementedError(
                "non-topologically-sorted data not supported",
            )
        selected = (
            alifestd_assign_contiguous_ids_polars(
                alifestd_collapse_unifurcations_polars(
                    selected, ignore_topological_sensitivity=True
                ),
            )
            .lazy()
            .collect()
        )
    selected = alifestd_mark_origin_time_delta_polars(
        selected, mark_as="__alifestd_tgs_branch_length"
    )

    trait_arrays = []
    for i, name in enumerate(trait_names):
        trait_mask = selected[f"__alifestd_tgs_trait{i}"]
        if trait_mask.has_nulls():
            raise NotImplementedError(
                f"null values in trait mask {name} not supported",
            )
        trait_arrays.append(trait_mask.to_numpy())

    prepared = TgsPrepared(
        ancestor_ids=selected["ancestor_id"].to_numpy(),
        origin_time=selected["origin_time"].to_numpy(),
        branch_lengths=selected["__alifestd_tgs_branch_length"].to_numpy(),
        trait_masks=trait_arrays,
        trait_names=list(trait_names),
    )
    return prepared, neutral_traits
