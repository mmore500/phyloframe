import logging
import typing

import numpy as np
import polars as pl

from .._auxlib._resolve_polars_expr import _resolve_polars_expr
from ._alifestd_calc_distance_matrix_asexual import (
    _alifestd_calc_distance_matrix_asexual_fast_path,
)
from ._alifestd_has_contiguous_ids_polars import (
    alifestd_has_contiguous_ids_polars,
)
from ._alifestd_is_topologically_sorted_polars import (
    alifestd_is_topologically_sorted_polars,
)
from ._alifestd_mark_csr_children_polars import (
    alifestd_mark_csr_children_polars,
)
from ._alifestd_mark_csr_offsets_polars import (
    alifestd_mark_csr_offsets_polars,
)
from ._alifestd_mark_num_children_polars import (
    alifestd_mark_num_children_polars,
)
from ._alifestd_try_add_ancestor_id_col_polars import (
    alifestd_try_add_ancestor_id_col_polars,
)


@_resolve_polars_expr("criterion")
def alifestd_calc_distance_matrix_polars(
    phylogeny_df: pl.DataFrame,
    *,
    criterion: typing.Union[str, pl.Expr] = "origin_time",
    progress_wrap: typing.Callable = lambda x: x,
) -> np.ndarray:
    """Calculate pairwise distances between all taxa via their MRCAs.

    The distance between two taxa is computed as the sum of criterion
    differences between each taxon and their Most Recent Common Ancestor
    (MRCA):

        distance[i, j] = (criterion[i] - criterion[mrca])
                       + (criterion[j] - criterion[mrca])

    Taxa sharing no common ancestor will have distance NaN.

    Pass tqdm or equivalent as `progress_wrap` to display a progress bar.

    Parameters
    ----------
    phylogeny_df : polars.DataFrame or polars.LazyFrame
        The phylogeny as a dataframe in alife standard format.

        Must represent an asexual phylogeny in working format (i.e.,
        topologically sorted with contiguous ids and an ancestor_id
        column, or an ancestor_list column from which ancestor_id can
        be derived).
    criterion : str or polars.Expr, default "origin_time"
        Column name or polars expression used to measure distance
        between taxa and their MRCA.
    progress_wrap : callable, optional
        Wrapper for progress display (e.g., tqdm).

    Returns
    -------
    numpy.ndarray
        Array of shape (n, n) with dtype float64, containing pairwise
        distances.  Entries are NaN where organisms share no common
        ancestor.

    See Also
    --------
    alifestd_calc_distance_matrix_asexual :
        Pandas-based implementation.
    alifestd_calc_mrca_id_matrix_asexual_polars :
        Computes the underlying MRCA id matrix.
    alifestd_find_pair_distance_polars :
        Computes distance for a single pair of taxa.
    """
    logging.info(
        "- alifestd_calc_distance_matrix_polars: " "adding ancestor_id col...",
    )
    phylogeny_df = alifestd_try_add_ancestor_id_col_polars(phylogeny_df)

    logging.info(
        "- alifestd_calc_distance_matrix_polars: "
        "checking contiguous ids...",
    )
    if not alifestd_has_contiguous_ids_polars(phylogeny_df):
        raise NotImplementedError(
            "non-contiguous ids not yet supported",
        )

    logging.info(
        "- alifestd_calc_distance_matrix_polars: "
        "checking topological sort...",
    )
    if not alifestd_is_topologically_sorted_polars(phylogeny_df):
        raise NotImplementedError(
            "topologically unsorted rows not yet supported",
        )

    schema_names = phylogeny_df.lazy().collect_schema().names()
    if "num_children" not in schema_names:
        logging.info(
            "- alifestd_calc_distance_matrix_polars: "
            "marking num_children...",
        )
        phylogeny_df = alifestd_mark_num_children_polars(phylogeny_df)
    if "csr_offsets" not in schema_names:
        logging.info(
            "- alifestd_calc_distance_matrix_polars: "
            "marking csr_offsets...",
        )
        phylogeny_df = alifestd_mark_csr_offsets_polars(phylogeny_df)
    if "csr_children" not in schema_names:
        logging.info(
            "- alifestd_calc_distance_matrix_polars: "
            "marking csr_children...",
        )
        phylogeny_df = alifestd_mark_csr_children_polars(phylogeny_df)

    logging.info(
        "- alifestd_calc_distance_matrix_polars: " "extracting arrays...",
    )
    ancestor_ids = (
        phylogeny_df.lazy()
        .select(pl.col("ancestor_id").cast(pl.Int64))
        .collect()
        .to_series()
        .to_numpy()
    )
    criterion_values = (
        phylogeny_df.lazy()
        .select(pl.col(criterion).cast(pl.Float64))
        .collect()
        .to_series()
        .to_numpy()
    )
    num_children = (
        phylogeny_df.lazy()
        .select("num_children")
        .collect()
        .to_series()
        .to_numpy()
    )
    csr_offsets = (
        phylogeny_df.lazy()
        .select(pl.col("csr_offsets").cast(pl.Int64))
        .collect()
        .to_series()
        .to_numpy()
    )
    csr_children = (
        phylogeny_df.lazy()
        .select(pl.col("csr_children").cast(pl.Int64))
        .collect()
        .to_series()
        .to_numpy()
    )

    logging.info(
        "- alifestd_calc_distance_matrix_polars: "
        "computing distance matrix...",
    )
    return _alifestd_calc_distance_matrix_asexual_fast_path(
        ancestor_ids,
        criterion_values,
        csr_offsets=csr_offsets,
        csr_children=csr_children,
        num_children=num_children,
    )
