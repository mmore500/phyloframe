import logging
import typing

import numpy as np
import polars as pl

from ._alifestd_calc_mrca_id_matrix_asexual import (
    _alifestd_calc_mrca_id_matrix_asexual_fast_path,
)
from ._alifestd_has_contiguous_ids_polars import (
    alifestd_has_contiguous_ids_polars,
)
from ._alifestd_is_topologically_sorted_polars import (
    alifestd_is_topologically_sorted_polars,
)
from ._alifestd_try_add_ancestor_id_col_polars import (
    alifestd_try_add_ancestor_id_col_polars,
)


def alifestd_calc_mrca_id_matrix_asexual_polars(
    phylogeny_df: pl.DataFrame,
    *,
    progress_wrap: typing.Callable = lambda x: x,
) -> np.ndarray:
    """Calculate the Most Recent Common Ancestor (MRCA) taxon id for each pair
    of taxa.

    Taxa sharing no common ancestor will have MRCA id -1.

    Parameters
    ----------
    phylogeny_df : polars.DataFrame or polars.LazyFrame
        The phylogeny as a dataframe in alife standard format.

        Must represent an asexual phylogeny in working format (i.e.,
        topologically sorted with contiguous ids and an ancestor_id
        column, or an ancestor_list column from which ancestor_id can
        be derived).
    progress_wrap : callable, optional
        Wrapper for progress display (e.g., tqdm).

    Returns
    -------
    numpy.ndarray
        Array of shape (n, n) with dtype int64, containing MRCA ids for
        each pair of organisms.  Entries are -1 where organisms share no
        common ancestor.

    See Also
    --------
    alifestd_calc_mrca_id_matrix_asexual :
        Pandas-based implementation.
    """
    logging.info(
        "- alifestd_calc_mrca_id_matrix_asexual_polars: "
        "adding ancestor_id col...",
    )
    phylogeny_df = alifestd_try_add_ancestor_id_col_polars(phylogeny_df)

    logging.info(
        "- alifestd_calc_mrca_id_matrix_asexual_polars: "
        "checking contiguous ids...",
    )
    if not alifestd_has_contiguous_ids_polars(phylogeny_df):
        raise NotImplementedError(
            "non-contiguous ids not yet supported",
        )

    logging.info(
        "- alifestd_calc_mrca_id_matrix_asexual_polars: "
        "checking topological sort...",
    )
    if not alifestd_is_topologically_sorted_polars(phylogeny_df):
        raise NotImplementedError(
            "topologically unsorted rows not yet supported",
        )

    logging.info(
        "- alifestd_calc_mrca_id_matrix_asexual_polars: "
        "extracting ancestor ids...",
    )
    ancestor_ids = (
        phylogeny_df.lazy()
        .select(pl.col("ancestor_id").cast(pl.Int64))
        .collect()
        .to_series()
        .to_numpy()
    )

    kwargs = {}
    schema_names = phylogeny_df.lazy().collect_schema().names()
    if "num_children" in schema_names:
        kwargs["num_children"] = (
            phylogeny_df.lazy()
            .select("num_children")
            .collect()
            .to_series()
            .to_numpy()
        )
    if "csr_offsets" in schema_names:
        kwargs["csr_offsets"] = (
            phylogeny_df.lazy()
            .select(pl.col("csr_offsets").cast(pl.Int64))
            .collect()
            .to_series()
            .to_numpy()
        )
    if "csr_children" in schema_names:
        kwargs["csr_children"] = (
            phylogeny_df.lazy()
            .select(pl.col("csr_children").cast(pl.Int64))
            .collect()
            .to_series()
            .to_numpy()
        )

    logging.info(
        "- alifestd_calc_mrca_id_matrix_asexual_polars: "
        "computing mrca id matrix...",
    )
    return _alifestd_calc_mrca_id_matrix_asexual_fast_path(
        ancestor_ids, **kwargs
    )
