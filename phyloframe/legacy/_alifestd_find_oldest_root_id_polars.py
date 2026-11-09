import typing

import polars as pl
import polars.selectors as cs

from ._alifestd_find_oldest_root_id_asexual import (
    _alifestd_find_oldest_root_id_asexual_fast_path,
)
from ._alifestd_try_add_ancestor_id_col_polars import (
    alifestd_try_add_ancestor_id_col_polars,
)


def alifestd_find_oldest_root_id_polars(
    phylogeny_df: pl.DataFrame,
) -> typing.Optional[int]:
    """Find the oldest root, measured by lowest `origin_time` (if available)
    or otherwise lowest `id`, breaking ties by lowest `id`.

    Parameters
    ----------
    phylogeny_df : polars.DataFrame or polars.LazyFrame
        The phylogeny as a dataframe in alife standard format.

        Must represent an asexual phylogeny.

    Returns
    -------
    int or None
        The id of the oldest root, or None if there are no roots.

    Warns
    -----
    UserWarning
        If a root's `origin_time` is null or nan; such roots sort last.

    See Also
    --------
    alifestd_find_oldest_root_id_asexual :
        Pandas-based implementation.
    """
    phylogeny_df = alifestd_try_add_ancestor_id_col_polars(phylogeny_df)
    roots_df = (
        phylogeny_df.lazy()
        .filter(pl.col("id") == pl.col("ancestor_id"))
        .select(
            cs.by_name("id", "ancestor_id"),
            cs.by_name("origin_time", require_all=False).cast(pl.Float64),
        )
        .collect()
    )
    has_origin_time = "origin_time" in roots_df.columns
    result = _alifestd_find_oldest_root_id_asexual_fast_path(
        roots_df["id"].to_numpy(),
        roots_df["ancestor_id"].to_numpy(),
        (
            roots_df["origin_time"].fill_null(float("nan")).to_numpy()
            if has_origin_time
            else None
        ),
    )
    return None if result == -1 else result
