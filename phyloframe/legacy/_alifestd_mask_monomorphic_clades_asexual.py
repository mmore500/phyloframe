import numpy as np
from packaging.version import parse
import pandas as pd

from .._auxlib._jit import jit
from ._alifestd_has_contiguous_ids import alifestd_has_contiguous_ids
from ._alifestd_is_topologically_sorted import alifestd_is_topologically_sorted
from ._alifestd_topological_sort import alifestd_topological_sort
from ._alifestd_try_add_ancestor_id_col import alifestd_try_add_ancestor_id_col


@jit(nopython=True)
def _alifestd_mask_monomorphic_clades_asexual_fast_path(
    ancestor_ids: np.ndarray,
    trait_mask: np.ndarray,
    trait_values: np.ndarray,
) -> np.ndarray:
    """Implementation detail for `alifestd_mask_monomorphic_clades_asexual`."""
    res = np.ones_like(trait_mask, dtype=np.bool_)

    for idx_r, ancestor_id in enumerate(ancestor_ids[::-1]):
        idx = len(ancestor_ids) - 1 - idx_r

        # handle genesis case
        if ancestor_id == idx:
            continue

        # clade has no defined trait values, so imposes no constraint
        if not trait_mask[idx]:
            continue

        if not trait_mask[ancestor_id]:
            trait_values[ancestor_id] = trait_values[idx]
            trait_mask[ancestor_id] = trait_mask[idx]

        res[ancestor_id] &= (
            res[idx] and trait_values[ancestor_id] == trait_values[idx]
        )

    return res


def _alifestd_mask_monomorphic_clades_asexual_slow_path(
    phylogeny_df: pd.DataFrame,
    mark_as: str,
) -> pd.DataFrame:
    """Implementation detail for `alifestd_mask_monomorphic_clades_asexual`."""
    phylogeny_df.index = phylogeny_df["id"]
    phylogeny_df[mark_as] = True

    for idx in reversed(phylogeny_df.index):
        ancestor_id = phylogeny_df.at[idx, "ancestor_id"]

        # handle genesis case
        if ancestor_id == idx:
            continue

        # clade has no defined trait values, so imposes no constraint
        if not phylogeny_df.at[
            idx, "alifestd_mask_monomorphic_clades_asexual_mask"
        ]:
            continue

        trait_value = phylogeny_df.at[
            idx, "alifestd_mask_monomorphic_clades_asexual_trait"
        ]
        ancestor_mask_value = phylogeny_df.at[
            ancestor_id, "alifestd_mask_monomorphic_clades_asexual_mask"
        ]

        if not ancestor_mask_value:
            phylogeny_df.at[
                ancestor_id,
                "alifestd_mask_monomorphic_clades_asexual_trait",
            ] = trait_value
            phylogeny_df.at[
                ancestor_id, "alifestd_mask_monomorphic_clades_asexual_mask"
            ] = phylogeny_df.at[
                idx, "alifestd_mask_monomorphic_clades_asexual_mask"
            ]

        result_value = (
            phylogeny_df.at[idx, mark_as]
            and phylogeny_df.at[
                ancestor_id,
                "alifestd_mask_monomorphic_clades_asexual_trait",
            ]
            == phylogeny_df.at[
                idx, "alifestd_mask_monomorphic_clades_asexual_trait"
            ]
        )

        phylogeny_df.at[ancestor_id, mark_as] &= result_value

    return phylogeny_df


def alifestd_mask_monomorphic_clades_asexual(
    phylogeny_df: pd.DataFrame,
    mutate: bool = False,
    *,
    mark_as: str = "alifestd_mask_monomorphic_clades_asexual",
    trait_mask: np.ndarray,
    trait_values: np.ndarray,
) -> pd.DataFrame:
    """
    Compute a mask marking "monomorphic" clades where all members with a trait
    defined value share the same trait value.

    Clades containing no members with a defined trait value are considered
    monomorphic. All leaf nodes are considered monomorphic.

    Parameters
    ----------
    phylogeny_df : pd.DataFrame
        DataFrame containing the phylogeny, including an `ancestor_id` column.
    mutate : bool, default=False
        If False, operates on a copy of `phylogeny_df`; if True, modifies
        `phylogeny_df` in place (but still returns it).
    mark_as : str, default "alifestd_mask_monomorphic_clades_asexual"
        Column name for the mask.
    trait_mask : np.ndarray
        Boolean array marking the nodes that have a defined trait value,
        aligned with `phylogeny_df.index`.
    trait_values : np.ndarray
        Array of trait values aligned with `phylogeny_df.index`.

    Returns
    -------
    pd.DataFrame
    """

    if not mutate:
        phylogeny_df = phylogeny_df.copy()

    phylogeny_df = alifestd_try_add_ancestor_id_col(phylogeny_df, mutate=True)

    # replace non-numeric dtypes with integer aliases
    if not np.issubdtype(trait_values.dtype, np.number):
        trait_values = pd.factorize(trait_values)[0]

    phylogeny_df["alifestd_mask_monomorphic_clades_asexual_mask"] = trait_mask
    phylogeny_df[
        "alifestd_mask_monomorphic_clades_asexual_trait"
    ] = trait_values

    if not alifestd_is_topologically_sorted(phylogeny_df):
        phylogeny_df = alifestd_topological_sort(phylogeny_df, mutate=True)

    if alifestd_has_contiguous_ids(phylogeny_df):
        phylogeny_df[
            mark_as
        ] = _alifestd_mask_monomorphic_clades_asexual_fast_path(
            phylogeny_df["ancestor_id"].to_numpy(),
            phylogeny_df[
                "alifestd_mask_monomorphic_clades_asexual_mask"
            ].to_numpy(
                dtype=np.bool_,
                copy=parse(pd.__version__) >= parse("3.0.0"),
            ),
            phylogeny_df[
                "alifestd_mask_monomorphic_clades_asexual_trait"
            ].to_numpy(
                copy=parse(pd.__version__) >= parse("3.0.0"),
            ),
        )
    else:
        phylogeny_df = _alifestd_mask_monomorphic_clades_asexual_slow_path(
            phylogeny_df, mark_as
        )

    phylogeny_df.drop(
        labels=[
            "alifestd_mask_monomorphic_clades_asexual_trait",
            "alifestd_mask_monomorphic_clades_asexual_mask",
        ],
        inplace=True,
        errors="ignore",
    )
    return phylogeny_df
