import numpy as np
import pandas as pd

from ._alifestd_is_working_format_asexual import (
    alifestd_is_working_format_asexual,
)
from ._alifestd_mark_leaves import _alifestd_mark_leaves_asexual_fast_path
from ._alifestd_mark_stem_node_id_asexual import (
    _alifestd_mark_stem_node_id_asexual_fast_path,
)
from ._alifestd_mask_trait_mdcs_strict_asexual import (
    _alifestd_mask_trait_uniform_clades_asexual_fast_path,
)
from ._alifestd_try_add_ancestor_id_col import alifestd_try_add_ancestor_id_col


def _alifestd_mask_trait_singletons_asexual_fast_path(
    ancestor_ids: np.ndarray,
    is_leaf: np.ndarray,
    trait_mask: np.ndarray,
) -> np.ndarray:
    """Implementation detail for
    `alifestd_mask_trait_singletons_asexual`."""
    is_trait_uniform = _alifestd_mask_trait_uniform_clades_asexual_fast_path(
        ancestor_ids, is_leaf, trait_mask
    )
    stem_node_ids = _alifestd_mark_stem_node_id_asexual_fast_path(ancestor_ids)
    # a leaf with the trait is a singleton unless every leaf of its stem
    # node's clade has the trait; a leaf without a stem node is its own stem
    # node, so is never a singleton
    return is_leaf & is_trait_uniform & ~is_trait_uniform[stem_node_ids]


def alifestd_mask_trait_singletons_asexual(
    phylogeny_df: pd.DataFrame,
    mutate: bool = False,
    *,
    mark_as: str = "alifestd_mask_trait_singletons_asexual",
    trait_mask: np.ndarray,
) -> pd.DataFrame:
    """Mark trait singletons: leaves that have a trait not shared by every
    leaf descending from their stem node, i.e., that gained it on their own
    pendant branch.

    A leaf's stem node is its nearest ancestor with at least two children,
    bypassing unifurcations (see `alifestd_mark_stem_node_id_asexual`).
    Leaves without the trait, and leaves without a stem node (e.g., root
    leaves), are marked False.

    Parameters
    ----------
    phylogeny_df : pd.DataFrame
        DataFrame containing the phylogeny, including an `ancestor_id` or
        `ancestor_list` column.
    mutate : bool, default False
        If False, operates on a copy of `phylogeny_df`; if True, modifies
        `phylogeny_df` in place (but still returns it).
    mark_as : str, default "alifestd_mask_trait_singletons_asexual"
        Column name for the mask.
    trait_mask : np.ndarray
        Boolean array marking nodes that have the trait, aligned with
        `phylogeny_df.index`, with boolean dtype. Values at inner nodes
        are ignored.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    NotImplementedError
        If `phylogeny_df` is not in working format (contiguous ids,
        topologically sorted; see `alifestd_to_working_format`), or if
        `trait_mask` does not have boolean dtype.
    """
    if not mutate:
        phylogeny_df = phylogeny_df.copy()

    phylogeny_df = alifestd_try_add_ancestor_id_col(phylogeny_df, mutate=True)
    trait_mask = np.asarray(trait_mask)
    if trait_mask.dtype != np.bool_:
        raise NotImplementedError(
            f"trait_mask must have boolean dtype, got {trait_mask.dtype}",
        )
    if not alifestd_is_working_format_asexual(phylogeny_df, mutate=True):
        raise NotImplementedError(
            "current implementation requires phylogeny_df in working format",
        )

    ancestor_ids = phylogeny_df["ancestor_id"].to_numpy()
    phylogeny_df[mark_as] = _alifestd_mask_trait_singletons_asexual_fast_path(
        ancestor_ids,
        (
            phylogeny_df["is_leaf"].to_numpy()
            if "is_leaf" in phylogeny_df.columns
            else _alifestd_mark_leaves_asexual_fast_path(ancestor_ids)
        ),
        trait_mask,
    )
    return phylogeny_df
