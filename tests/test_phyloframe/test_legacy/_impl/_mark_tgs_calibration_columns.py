import typing

import polars as pl

from phyloframe.legacy import (
    alifestd_mark_clade_size_geom_classes_polars,
    alifestd_mark_joint_quantile_bins_polars,
    alifestd_mark_leaves_polars,
    alifestd_mark_origin_time_delta_polars,
)


def mark_tgs_calibration_columns(
    phylogeny_df: pl.DataFrame,
    *,
    neutral_traits: typing.Sequence[str],
    size_ratio: float = 2.0,
    min_class_weight: float = 20.0,
    max_num_bins: int = 4096,
) -> pl.DataFrame:
    """Mark calibration partitions, `clade_size_geom_class`, and bins within
    them, `joint_quantile_bin`, as the tgs estimators' Examples show."""
    phylogeny_df = alifestd_mark_leaves_polars(
        alifestd_mark_origin_time_delta_polars(
            phylogeny_df, mark_as="branch_length"
        )
    )
    phylogeny_df = alifestd_mark_clade_size_geom_classes_polars(
        phylogeny_df,
        size_ratio=size_ratio,
        weight_by=list(dict.fromkeys(neutral_traits)),
        min_class_weight=min_class_weight,
    )
    return alifestd_mark_joint_quantile_bins_polars(
        phylogeny_df,
        bin_by=["branch_length", "origin_time"],
        partition_by="clade_size_geom_class",
        max_num_bins=max_num_bins,
    )
