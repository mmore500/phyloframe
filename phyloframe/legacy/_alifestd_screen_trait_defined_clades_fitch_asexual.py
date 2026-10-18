import logging
import typing

import numpy as np
from packaging.version import parse
import pandas as pd

from .._auxlib._jit import jit
from ._alifestd_has_contiguous_ids import alifestd_has_contiguous_ids
from ._alifestd_is_topologically_sorted import alifestd_is_topologically_sorted
from ._alifestd_is_working_format_asexual import (
    alifestd_is_working_format_asexual,
)
from ._alifestd_mark_node_depth_asexual import alifestd_mark_node_depth_asexual
from ._alifestd_topological_sort import alifestd_topological_sort
from ._alifestd_try_add_ancestor_id_col import alifestd_try_add_ancestor_id_col


@jit(nopython=True)
def _alifestd_screen_trait_defined_clades_fitch_asexual_fast_path(
    ancestor_ids: np.ndarray,
    ft_set: np.ndarray,
) -> np.ndarray:
    """Implementation detail for
    `alifestd_screen_trait_defined_clades_fitch_asexual`."""
    num_nodes = len(ancestor_ids)
    ft_intersect = np.full(num_nodes, 3, dtype=np.uint8)
    ft_union = np.zeros(num_nodes, dtype=np.uint8)
    ft_num0s = np.zeros(num_nodes, dtype=np.int64)
    ft_num1s = np.zeros(num_nodes, dtype=np.int64)

    # bottom-up pass: iterate back to front, so children are resolved before
    # parents, accumulating children's trait set statistics at parents
    for idx_r, ancestor_id in enumerate(ancestor_ids[::-1]):
        idx = num_nodes - 1 - idx_r
        if ft_set[idx] == 0:
            # if majority of children have trait absent/present, set that
            if ft_num0s[idx] > ft_num1s[idx]:
                ft_set[idx] = 1
            elif ft_num1s[idx] > ft_num0s[idx]:
                ft_set[idx] = 2
            else:
                # trait set is intersection if nonempty, else union
                ft_set[idx] = ft_intersect[idx] or ft_union[idx]

        if ancestor_id != idx:  # exclude genesis cases
            ft_num0s[ancestor_id] += ft_set[idx] & 1
            ft_num1s[ancestor_id] += ft_set[idx] >> 1
            ft_intersect[ancestor_id] &= ft_set[idx]
            ft_union[ancestor_id] |= ft_set[idx]

    # top-down pass: iterate front to back, so parents are resolved before
    # children, resolving ambiguity in favor of parent's trait state
    res = np.zeros(num_nodes, dtype=np.bool_)
    for idx, ancestor_id in enumerate(ancestor_ids):
        if ancestor_id == idx:  # handle genesis cases
            if ft_set[idx] == 3:
                ft_set[idx] = 1  # root assumed to be trait absent
            continue  # mutation at root not considered a change

        # inherit shared trait
        common_trait = ft_set[idx] & ft_set[ancestor_id]
        if common_trait != 0 or ft_set[idx] == 0:
            ft_set[idx] = common_trait

        res[idx] = ft_set[idx] == 2 and ft_set[ancestor_id] != 2

    return res


def _alifestd_screen_trait_defined_clades_fitch_asexual_slow_path(
    phylogeny_df: pd.DataFrame,
    ft_set: np.ndarray,
    progress_wrap: typing.Callable,
) -> np.ndarray:
    """Implementation detail for
    `alifestd_screen_trait_defined_clades_fitch_asexual`."""
    phylogeny_df["fitch_trait_set"] = ft_set.astype(int)

    saved_id_order = None
    if not alifestd_is_topologically_sorted(phylogeny_df):
        saved_id_order = phylogeny_df["id"].to_numpy(copy=True)
        phylogeny_df = alifestd_topological_sort(phylogeny_df, mutate=True)

    if "node_depth" not in phylogeny_df.columns:
        phylogeny_df = alifestd_mark_node_depth_asexual(
            phylogeny_df, mutate=True
        )
    max_depth = phylogeny_df["node_depth"].max()

    if alifestd_has_contiguous_ids(phylogeny_df):
        ancestor_iloc = phylogeny_df["ancestor_id"].values
    else:
        phylogeny_df.set_index("id", drop=False, inplace=True)
        ancestor_iloc = phylogeny_df.index.get_indexer(
            phylogeny_df["ancestor_id"].values,
        )

    phylogeny_df["fitch_trait_intersect"] = 0

    ft_intersect = phylogeny_df["fitch_trait_intersect"].to_numpy(
        copy=parse(pd.__version__) >= parse("3.0.0"),
    )
    ft_set = phylogeny_df["fitch_trait_set"].to_numpy(
        copy=parse(pd.__version__) >= parse("3.0.0"),
    )
    ft_union = ft_set.copy()
    node_depth = phylogeny_df["node_depth"].to_numpy(copy=False)

    logging.info(
        " - alifestd_screen_trait_defined_clades_fitch_asexual: "
        "performing bottom-up pass...",
    )
    # process deepest level first, so children are resolved before parents
    for depth in progress_wrap(reversed(range(max_depth + 1))):
        ft_intersect[:] = 3
        # at each parent, calculate intersection of children's trait sets
        np.bitwise_and.at(
            ft_intersect,
            ancestor_iloc,
            np.where(node_depth == 0, 3, ft_set),  # handle roof self-loop
        )

        # at each parent, calculate counts of children's trait sets
        ft_num0s = np.zeros(len(phylogeny_df), dtype=int)
        ft_num1s = np.zeros(len(phylogeny_df), dtype=int)
        np.add.at(ft_num0s, ancestor_iloc, (ft_set & 1))
        np.add.at(ft_num1s, ancestor_iloc, (ft_set & 2) >> 1)

        # at each parent, calculate union of children's trait sets
        np.bitwise_or.at(ft_union, ancestor_iloc, ft_set)

        # if majority of children have trait absent/present, set that trait
        proposed_update = np.where(
            ft_num0s == ft_num1s,
            # trait set is intersection if nonempty, else union
            np.where(ft_intersect == 0, ft_union, ft_intersect),
            (ft_num0s > ft_num1s) + 2 * (ft_num1s > ft_num0s),
        )

        # only update trait set if not already set, for current level only
        ft_set = np.where(
            (ft_set == 0) & (node_depth == depth),
            proposed_update,
            ft_set,
        )

    logging.info(
        " - alifestd_screen_trait_defined_clades_fitch_asexual: "
        "performing top-down pass...",
    )
    default_root = 1  # root assumed to be trait absent
    ambiguous_root_mask = (node_depth == 0) & (ft_set == 3)
    ft_set[ambiguous_root_mask] = default_root
    for depth in progress_wrap(range(1, max_depth + 1)):
        layer_mask = node_depth == depth
        traitless_mask = ft_set == 0
        common_trait = ft_set & ft_set[ancestor_iloc]
        # inherit shared trait
        inherit_mask = layer_mask & (
            common_trait.astype(bool) | traitless_mask
        )
        ft_set[inherit_mask] = common_trait[inherit_mask]
        assert (np.bitwise_count(ft_set[layer_mask]) == 1).all()

    # finalization
    ft_set >>= 1  # map {1, 2} to {False, True}
    phylogeny_df["alifestd_screen_trait_defined_clades_fitch_asexual"] = (
        # root self-loop ok: mutation at root not considered a change
        (~ft_set[ancestor_iloc])
        & ft_set
    ).astype(bool)
    phylogeny_df.drop(
        columns=["fitch_trait_set", "fitch_trait_intersect"],
        inplace=True,
    )
    if saved_id_order is not None:
        phylogeny_df.index = phylogeny_df["id"]
        return phylogeny_df.loc[
            saved_id_order,
            "alifestd_screen_trait_defined_clades_fitch_asexual",
        ].to_numpy()
    else:
        return phylogeny_df[
            "alifestd_screen_trait_defined_clades_fitch_asexual"
        ].values


def alifestd_screen_trait_defined_clades_fitch_asexual(
    phylogeny_df: pd.DataFrame,
    mutate: bool = False,
    *,
    mask_trait_absent: np.ndarray,  # 1D boolean array
    mask_trait_present: np.ndarray,  # 1D boolean array
    progress_wrap: typing.Callable = lambda x: x,
) -> np.ndarray:  # 1D boolean array
    """Perform a maximum parsimony screen for trait-defined clades using
    Fitch's algorithm.

    The `mask_trait_absent` parameter can be used to exclude nodes from
    consideration, for instance &'ing with `alifestd_mark_leaves` can be used
    to only consider traits on leaves.

    Pass tqdm or equivalent as `progress_wrap` to display a progress bar.

    Default root state is assumed to be False.

    Returns a numpy array of bool with the same length as the input DataFrame,
    with array elements as the number of nodes in the clade that have the
    trait. Returned array matches row order of the input DataFrame.
    """
    if len(phylogeny_df) == 0:
        return np.array([], dtype=bool)

    mask_trait_absent = np.asarray(mask_trait_absent, dtype=bool)
    mask_trait_present = np.asarray(mask_trait_present, dtype=bool)
    if (mask_trait_absent & mask_trait_present).any():
        raise ValueError("conflicting trait presence/absence masks")

    if not mutate:
        phylogeny_df = phylogeny_df.copy()

    phylogeny_df = alifestd_try_add_ancestor_id_col(phylogeny_df, mutate=True)
    ft_set = (mask_trait_absent + 2 * mask_trait_present).astype(np.uint8)
    if alifestd_is_working_format_asexual(phylogeny_df, mutate=True):
        return _alifestd_screen_trait_defined_clades_fitch_asexual_fast_path(
            pd.to_numeric(phylogeny_df["ancestor_id"]).to_numpy(),
            ft_set,
        )
    else:
        return _alifestd_screen_trait_defined_clades_fitch_asexual_slow_path(
            phylogeny_df,
            ft_set,
            progress_wrap=progress_wrap,
        )
