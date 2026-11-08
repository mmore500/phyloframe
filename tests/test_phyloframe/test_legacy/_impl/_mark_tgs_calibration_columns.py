import typing

import polars as pl

from phyloframe.legacy import (
    alifestd_mark_clade_size_classes_polars,
    alifestd_mark_joint_quantile_bins_polars,
    alifestd_mark_leaves_polars,
    alifestd_mark_stem_node_id_polars,
)


def mark_tgs_calibration_columns(
    phylogeny_df: pl.DataFrame,
    *,
    neutral_traits: typing.Sequence[str],
    size_ratio: float = 2.0,
    min_class_weight: float = 20.0,
    max_num_bins: int = 4096,
) -> pl.DataFrame:
    """Mark calibration partitions, `clade_size_class`, and bins within
    them, `joint_quantile_bin`, as the tgs estimators' Notes describe."""
    phylogeny_df = alifestd_mark_leaves_polars(
        alifestd_mark_stem_node_id_polars(phylogeny_df)
    ).with_columns(
        branch_length=pl.col("origin_time")
        - pl.col("origin_time").gather(pl.col("stem_node_id")),
        # neutral gains; a root's transition is its initial state
        neutral_gains=pl.when(pl.col("ancestor_id") != pl.col("id"))
        .then(pl.sum_horizontal(list(dict.fromkeys(neutral_traits))))
        .otherwise(0),
    )
    phylogeny_df = alifestd_mark_clade_size_classes_polars(
        phylogeny_df,
        size_ratio=size_ratio,
        weight_by=["neutral_gains"],
        min_class_weight=min_class_weight,
    )
    return alifestd_mark_joint_quantile_bins_polars(
        phylogeny_df,
        bin_by=["branch_length", "origin_time"],
        partition_by="clade_size_class",
        max_num_bins=max_num_bins,
    )
