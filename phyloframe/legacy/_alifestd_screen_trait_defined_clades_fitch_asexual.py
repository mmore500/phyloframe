import logging
import typing

import numpy as np
import pandas as pd

from ._alifestd_has_contiguous_ids import alifestd_has_contiguous_ids
from ._alifestd_mark_node_depth_asexual import alifestd_mark_node_depth_asexual
from ._alifestd_try_add_ancestor_id_col import alifestd_try_add_ancestor_id_col


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

    Nodes are resolved one depth level at a time in the bottom-up pass,
    deepest first, so each node's trait set is computed only after all of its
    children are resolved. Multifurcations use the generalized Fitch rule,
    taking the trait(s) most common among children. Nodes in neither mask
    are inferred; leaves in neither mask do not constrain their parent.

    Pass tqdm or equivalent as `progress_wrap` to display a progress bar.

    Default root state is assumed to be False.

    Returns a numpy array of bool with the same length as the input DataFrame,
    with array elements True for nodes where the trait is gained (i.e., nodes
    with trait present whose parent has trait absent). Returned array matches
    row order of the input DataFrame.
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
    # track input row order, which node depth calculation may not preserve
    phylogeny_df["fitch_input_row"] = np.arange(len(phylogeny_df))
    # trait sets as bit flags: 1 is trait absent, 2 is trait present
    phylogeny_df["fitch_trait_set"] = (
        mask_trait_absent + 2 * mask_trait_present
    )

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

    # retrieve trait sets after any reordering by node depth calculation
    ft_set = phylogeny_df["fitch_trait_set"].to_numpy(copy=True)
    node_depth = phylogeny_df["node_depth"].to_numpy(copy=False)

    # group node indices by depth level
    depth_order = np.argsort(node_depth, kind="stable")
    level_bounds = np.searchsorted(
        node_depth[depth_order], np.arange(max_depth + 2)
    )

    logging.info(
        " - alifestd_screen_trait_defined_clades_fitch_asexual: "
        "performing bottom-up pass...",
    )
    # process deepest level first, so children are resolved before parents
    for depth in progress_wrap(range(max_depth, 0, -1)):
        children = depth_order[level_bounds[depth] : level_bounds[depth + 1]]
        parents, inverse = np.unique(
            ancestor_iloc[children], return_inverse=True
        )
        child_sets = ft_set[children]

        # at each parent, calculate counts of children's trait sets
        ft_num0s = np.bincount(
            inverse, weights=child_sets & 1, minlength=len(parents)
        )
        ft_num1s = np.bincount(
            inverse, weights=(child_sets & 2) >> 1, minlength=len(parents)
        )

        # trait set is the most common trait(s) among children; if no child
        # is constrained, both counts are zero and the set is ambiguous
        proposed_update = (ft_num0s >= ft_num1s) + 2 * (ft_num1s >= ft_num0s)

        # only update trait set if not already set
        unset_mask = ft_set[parents] == 0
        ft_set[parents[unset_mask]] = proposed_update[unset_mask]

    # remaining unset nodes (unobserved leaves) are unconstrained
    ft_set[ft_set == 0] = 3

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
    result = np.empty(len(phylogeny_df), dtype=bool)
    result[phylogeny_df["fitch_input_row"].to_numpy()] = phylogeny_df[
        "alifestd_screen_trait_defined_clades_fitch_asexual"
    ].to_numpy()
    phylogeny_df.drop(
        columns=["fitch_trait_set", "fitch_input_row"],
        inplace=True,
    )
    return result
