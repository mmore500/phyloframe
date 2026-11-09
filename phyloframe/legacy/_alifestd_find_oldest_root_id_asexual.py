import typing

import numpy as np
import pandas as pd

from ._alifestd_try_add_ancestor_id_col import alifestd_try_add_ancestor_id_col


def _alifestd_find_oldest_root_id_asexual_fast_path(
    ids: np.ndarray,
    ancestor_ids: np.ndarray,
    origin_times: typing.Optional[np.ndarray],
) -> int:
    """Find the oldest root of an asexual phylogeny.

    Parameters
    ----------
    ids : np.ndarray
        Array of taxon ids.
    ancestor_ids : np.ndarray
        Array of ancestor ids, aligned with `ids`; roots are their own
        ancestors.
    origin_times : np.ndarray, optional
        Array of origin times, aligned with `ids`; if None, the oldest root
        is the lowest-id root.

    Returns
    -------
    int
        The id of the root with lowest origin time, breaking ties by lowest
        id, or -1 if there are no roots.

    Raises
    ------
    NotImplementedError
        If a root's origin time is nan, as then which is oldest is
        undefined.
    """
    is_root = ids == ancestor_ids
    root_ids = ids[is_root]
    if not len(root_ids):
        return -1
    if origin_times is None:
        return int(root_ids.min())
    root_origin_times = origin_times[is_root]
    if np.isnan(root_origin_times).any():
        raise NotImplementedError(
            "null or nan root origin_time not supported, as then which root "
            "is oldest is undefined",
        )
    return int(root_ids[np.lexsort((root_ids, root_origin_times))[0]])


def alifestd_find_oldest_root_id_asexual(
    phylogeny_df: pd.DataFrame,
    *,
    mutate: bool = False,
) -> typing.Optional[int]:
    """Find the oldest root, measured by lowest `origin_time` (if available)
    or otherwise lowest `id`, breaking ties by lowest `id`.

    Parameters
    ----------
    phylogeny_df : pd.DataFrame
        Phylogeny in alife standard format.

        Must represent an asexual phylogeny.
    mutate : bool, default False
        If True, allows in-place modification of `phylogeny_df`.

    Returns
    -------
    int or None
        The id of the oldest root, or None if there are no roots.

    Raises
    ------
    NotImplementedError
        If a root's `origin_time` is null or nan.

    See Also
    --------
    alifestd_find_oldest_root_id_polars :
        Polars-based implementation.
    alifestd_mark_oldest_root :
        Marks the oldest root.
    """
    phylogeny_df = alifestd_try_add_ancestor_id_col(
        phylogeny_df, mutate=mutate
    )
    result = _alifestd_find_oldest_root_id_asexual_fast_path(
        phylogeny_df["id"].to_numpy(),
        phylogeny_df["ancestor_id"].to_numpy(),
        (
            phylogeny_df["origin_time"].to_numpy(dtype=float, na_value=np.nan)
            if "origin_time" in phylogeny_df
            else None
        ),
    )
    return None if result == -1 else result
