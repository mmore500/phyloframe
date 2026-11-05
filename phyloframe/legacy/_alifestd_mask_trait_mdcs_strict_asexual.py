import numpy as np
import pandas as pd

from ._alifestd_is_working_format_asexual import (
    alifestd_is_working_format_asexual,
)
from ._alifestd_mark_leaves import _alifestd_mark_leaves_asexual_fast_path
from ._alifestd_mark_lineage_delta_asexual import (
    _alifestd_mark_lineage_delta_asexual_fast_path,
)
from ._alifestd_mask_monomorphic_clades_asexual import (
    _alifestd_mask_monomorphic_clades_asexual_fast_path,
)
from ._alifestd_try_add_ancestor_id_col import alifestd_try_add_ancestor_id_col


def _alifestd_mask_trait_mdcs_strict_asexual_clade_fast_path(
    ancestor_ids: np.ndarray,
    trait_mask: np.ndarray,
) -> np.ndarray:
    """Implementation detail for `alifestd_mask_trait_mdcs_strict_asexual`.

    Returns, for each node, whether every leaf of its clade has the trait.
    Assumes contiguous, topologically sorted ids.
    """
    is_leaf = _alifestd_mark_leaves_asexual_fast_path(ancestor_ids)
    # carries each clade's leaf value up to its inner nodes, so that a
    # monomorphic clade's representative value is its leaves' shared value
    representative = is_leaf & np.asarray(trait_mask, dtype=bool)
    is_monomorphic = _alifestd_mask_monomorphic_clades_asexual_fast_path(
        ancestor_ids, is_leaf.copy(), representative
    )
    return is_monomorphic & representative


def _alifestd_mask_trait_mdcs_strict_asexual_fast_path(
    ancestor_ids: np.ndarray,
    trait_mask: np.ndarray,
) -> np.ndarray:
    """Implementation detail for `alifestd_mask_trait_mdcs_strict_asexual`."""
    is_all_carrier = _alifestd_mask_trait_mdcs_strict_asexual_clade_fast_path(
        ancestor_ids, trait_mask
    )
    # an MDC's root is where the all-carrier clade mask turns on along a
    # lineage, i.e., where its lineage delta is +1; a root's delta is its
    # own value, but a root has no stem branch, so is excluded
    is_root = ancestor_ids == np.arange(len(ancestor_ids))
    return (
        _alifestd_mark_lineage_delta_asexual_fast_path(
            ancestor_ids, is_all_carrier
        )
        == 1
    ) & ~is_root


def alifestd_mask_trait_mdcs_strict_asexual(
    phylogeny_df: pd.DataFrame,
    mutate: bool = False,
    *,
    mark_as: str = "alifestd_mask_trait_mdcs_strict_asexual",
    trait_mask: np.ndarray,
) -> pd.DataFrame:
    """Mark the root of each mutationally defined clade (MDC): a maximal
    clade whose leaves all have a trait, i.e., a clade whose every leaf has
    the trait but whose parent's clade does not.

    Under parsimony without loss, an MDC gained the trait on the branch
    leading to its root, its stem branch. Single-leaf MDCs are trait
    singletons (see `alifestd_mask_trait_singletons_asexual`); larger MDCs
    are clades sharing one gain. Roots are marked False, as they have no
    stem branch to have gained the trait on. A unifurcation's clade has the
    same leaves as its child's, so along a chain of unifurcations, only the
    rootmost node is marked.

    Strict, as a clade qualifies only if every one of its leaves has the
    trait. Builds on `alifestd_mask_monomorphic_clades_asexual`, which
    marks clades whose leaves share a value, but excludes clades sharing
    absence of the trait and marks only each maximal clade's root.

    Parameters
    ----------
    phylogeny_df : pd.DataFrame
        DataFrame containing the phylogeny, including an `ancestor_id` or
        `ancestor_list` column.
    mutate : bool, default False
        If False, operates on a copy of `phylogeny_df`; if True, modifies
        `phylogeny_df` in place (but still returns it).
    mark_as : str, default "alifestd_mask_trait_mdcs_strict_asexual"
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

    See Also
    --------
    alifestd_mask_monomorphic_clades_asexual :
        Marks every node whose clade's leaves share a trait value, present
        or absent; this function marks only the rootmost node of each clade
        whose leaves all have the trait.
    alifestd_mask_trait_mdcs_strict_polars :
        Polars-based implementation.
    alifestd_mask_trait_singletons_asexual :
        Marks single-leaf MDCs only, bypassing unifurcations.
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

    phylogeny_df[mark_as] = _alifestd_mask_trait_mdcs_strict_asexual_fast_path(
        phylogeny_df["ancestor_id"].to_numpy(),
        trait_mask,
    )
    return phylogeny_df
