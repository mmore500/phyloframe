import polars as pl

from ._alifestd_has_contiguous_ids_polars import (
    alifestd_has_contiguous_ids_polars,
)
from ._alifestd_is_topologically_sorted_polars import (
    alifestd_is_topologically_sorted_polars,
)


def alifestd_is_working_format_polars(
    phylogeny_df: pl.DataFrame,
) -> bool:
    """Test if phylogeny_df is an asexual phylogeny in working format.

    The working format is a dataframe with the following properties:
      - contains an integer datatype `ancestor_id` column,
      - topologically sorted (organisms appear after all ancestors), and
      - contiguous ids (organisms' ids correspond to row number).
    """
    if "ancestor_id" not in phylogeny_df.lazy().collect_schema().names():
        return False

    return alifestd_is_topologically_sorted_polars(
        phylogeny_df,
    ) and alifestd_has_contiguous_ids_polars(phylogeny_df)
