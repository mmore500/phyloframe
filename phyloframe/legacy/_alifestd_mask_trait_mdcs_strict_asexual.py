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


def _alifestd_mask_trait_uniform_clades_asexual_fast_path(
    ancestor_ids: np.ndarray,
    is_leaf: np.ndarray,
    trait_mask: np.ndarray,
) -> np.ndarray:
    """Implementation detail for `alifestd_mask_trait_mdcs_strict_asexual`
    and `alifestd_mask_trait_singletons_asexual`.

    Returns, for each node, whether every leaf of its clade has the trait.
    Assumes contiguous, topologically sorted ids.
    """
    # carries each clade's leaf value up to its inner nodes, so that a
    # monomorphic clade's representative value is its leaves' shared value
    representative = is_leaf & np.asarray(trait_mask, dtype=bool)
    is_monomorphic = _alifestd_mask_monomorphic_clades_asexual_fast_path(
        ancestor_ids, np.array(is_leaf, dtype=bool), representative
    )
    return is_monomorphic & representative


def _alifestd_mask_trait_mdcs_strict_asexual_fast_path(
    ancestor_ids: np.ndarray,
    is_leaf: np.ndarray,
    trait_mask: np.ndarray,
) -> np.ndarray:
    """Implementation detail for `alifestd_mask_trait_mdcs_strict_asexual`."""
    is_trait_uniform = _alifestd_mask_trait_uniform_clades_asexual_fast_path(
        ancestor_ids, is_leaf, trait_mask
    )
    # an MDC's root is where clades whose leaves all have the trait begin
    # along a lineage, i.e., where the lineage delta is +1; roots have no
    # stem branch, so are excluded
    is_root = ancestor_ids == np.arange(len(ancestor_ids))
    return (
        _alifestd_mark_lineage_delta_asexual_fast_path(
            ancestor_ids, is_trait_uniform
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

    Roots are marked False, as they have no stem branch to have gained the
    trait on. A unifurcation's clade has the same leaves as its child's, so
    along a chain of unifurcations, only the earliest node is marked.

    Strict, as a clade qualifies only if every one of its leaves has the
    trait.

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

    ancestor_ids = phylogeny_df["ancestor_id"].to_numpy()
    phylogeny_df[mark_as] = _alifestd_mask_trait_mdcs_strict_asexual_fast_path(
        ancestor_ids,
        (
            phylogeny_df["is_leaf"].to_numpy()
            if "is_leaf" in phylogeny_df.columns
            else _alifestd_mark_leaves_asexual_fast_path(ancestor_ids)
        ),
        trait_mask,
    )
    return phylogeny_df
