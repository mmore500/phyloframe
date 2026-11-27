import numpy as np
import pandas as pd

from ._alifestd_is_working_format_asexual import (
    alifestd_is_working_format_asexual,
)
from ._alifestd_mark_lineage_cummax_asexual import (
    _alifestd_mark_lineage_cummax_asexual_fast_path,
)
from ._alifestd_mark_lineage_cummin_asexual import (
    _alifestd_mark_lineage_cummin_asexual_fast_path,
)
from ._alifestd_try_add_ancestor_id_col import alifestd_try_add_ancestor_id_col


def _alifestd_mask_persistence_asexual_fast_path(
    ancestor_ids: np.ndarray,
    extant_mask: np.ndarray,
    candidate_mask: np.ndarray,
    criterion_values: np.ndarray,
) -> np.ndarray:
    """Implementation detail for `alifestd_mask_persistence_asexual`
    and `alifestd_mask_persistence_polars`.

    Assumes contiguous, topologically sorted ids, with finite float
    `criterion_values` that do not decrease from ancestor to descendant.
    """
    # earliest candidate in each node's clade, the clade's candidate closest
    # to any taxon outside the clade
    clade_candidate_values = _alifestd_mark_lineage_cummin_asexual_fast_path(
        ancestor_ids,
        np.where(candidate_mask, criterion_values, np.nan),
        True,  # reverse
        True,  # skipna
    )
    # distance via an mrca is criterion[extant] + criterion[candidate]
    # - 2 * criterion[mrca]; scoring each node as an mrca, by the terms that
    # do not depend on the extant taxon, the lowest score along a taxon's
    # lineage is at its mrca with its closest candidates (where a clade's
    # earliest candidate shares a later mrca with the taxon, the score only
    # overestimates distance, as criterion values do not decrease)
    mrca_scores = clade_candidate_values - 2 * criterion_values
    best_scores = _alifestd_mark_lineage_cummin_asexual_fast_path(
        ancestor_ids, mrca_scores, False, True
    )
    # a node is mrca of an extant taxon and its closest candidates if its
    # score is the best along that taxon's lineage; no taxon in a node's
    # clade can have a best score exceeding the node's own score
    is_closest_mrca = mrca_scores == (
        _alifestd_mark_lineage_cummax_asexual_fast_path(
            ancestor_ids,
            np.where(extant_mask, best_scores, np.nan),
            True,  # reverse
            True,  # skipna
        )
    )
    # a candidate is a closest relative if it is the earliest candidate in
    # the clade of such an mrca; no ancestor's clade can have an earliest
    # candidate later than the candidate itself
    return candidate_mask & (
        criterion_values
        == _alifestd_mark_lineage_cummax_asexual_fast_path(
            ancestor_ids,
            np.where(is_closest_mrca, clade_candidate_values, np.nan),
            False,  # reverse
            True,  # skipna
        )
    )


def alifestd_mask_persistence_asexual(
    phylogeny_df: pd.DataFrame,
    mutate: bool = False,
    *,
    candidate_mask: np.ndarray,
    criterion: str = "origin_time",
    extant_mask: np.ndarray,
    mark_as: str = "alifestd_mask_persistence_asexual",
) -> pd.DataFrame:
    """Mark candidate taxa that are the closest relative, among candidates,
    of at least one extant taxon.

    Useful to identify which earlier-sampled taxa (candidates) have
    lineages that persist among later-sampled taxa (extant), where
    candidates may not be direct ancestors of extant taxa (e.g., both are
    leaves).

    Distance between an extant taxon and a candidate is measured via their
    most recent common ancestor (MRCA), as in
    `alifestd_find_pair_distance_asexual`:

        distance = (criterion[extant] - criterion[mrca])
                 + (criterion[candidate] - criterion[mrca])

    If several candidates tie as an extant taxon's closest, all are marked.
    Extant taxa that share no common ancestor with any candidate do not
    mark any candidate. A taxon that is both extant and a candidate is its
    own closest relative.

    Parameters
    ----------
    phylogeny_df : pd.DataFrame
        DataFrame containing the phylogeny, including an `ancestor_id` or
        `ancestor_list` column.
    mutate : bool, default False
        If False, operates on a copy of `phylogeny_df`; if True, modifies
        `phylogeny_df` in place (but still returns it).
    candidate_mask : np.ndarray
        Boolean array marking taxa eligible to be marked as closest
        relatives, e.g., leaves within an earlier time window, aligned with
        `phylogeny_df.index`, with boolean dtype. May include inner nodes.
    criterion : str, default "origin_time"
        Column name used to measure distance between taxa and their MRCA,
        e.g., "origin_time" for elapsed time or "node_depth" for number of
        edges (see `alifestd_mark_node_depth_asexual`). Values must be
        finite, and must not decrease from ancestor to descendant.
    extant_mask : np.ndarray
        Boolean array marking taxa whose closest relatives are sought, e.g.,
        leaves after a time threshold, aligned with `phylogeny_df.index`,
        with boolean dtype. May include inner nodes.
    mark_as : str, default "alifestd_mask_persistence_asexual"
        Column name for the mask.

    Returns
    -------
    pd.DataFrame
        The phylogeny with an added boolean mask column, a subset of
        `candidate_mask`.

    Raises
    ------
    NotImplementedError
        If `phylogeny_df` is not in working format (contiguous ids,
        topologically sorted; see `alifestd_to_working_format`), if
        `extant_mask` or `candidate_mask` does not have boolean dtype, or if
        `criterion` values are non-finite or decrease from ancestor to
        descendant (see `alifestd_coerce_chronological_consistency`).
    ValueError
        If `criterion` is not a column in `phylogeny_df`.

    See Also
    --------
    alifestd_find_pair_distance_asexual :
        Calculates distance between a single pair of taxa.
    alifestd_mark_sample_tips_lineage_asexual :
        Marks leaves closest to the lineage of a single target leaf.
    alifestd_mask_persistence_polars :
        Polars-based implementation.
    alifestd_prune_extinct_lineages_asexual :
        Drops taxa without marked descendants, e.g., pass `mark_as` column
        as `criterion` to keep only lineages of marked candidates.
    alifestd_ultrametricize :
        Makes leaves contemporaneous, e.g., to mark candidates by most
        recent common ancestor.

    Notes
    -----
    Which candidates are closest depends on `criterion`. With leaves'
    own "origin_time" values, a candidate on a long branch can be further
    from an extant taxon than a candidate that diverged from its lineage
    earlier. With contemporaneous leaves, an extant leaf's closest candidate
    leaves are those sharing its most recent common ancestor with any
    candidate; see Examples.

    Examples
    --------
    Take phylogeny `df` in working format. To mark leaves from times 40 to
    60 with lineages that persist among leaves after time 100:

    >>> is_leaf = alifestd_mark_leaves(df)["is_leaf"].to_numpy()
    >>> origin_time = df["origin_time"].to_numpy()
    >>> candidate_mask = is_leaf & (origin_time >= 40) & (origin_time <= 60)
    >>> extant_mask = is_leaf & (origin_time > 100)
    >>> persistence_df = alifestd_mask_persistence_asexual(
    ...     df,
    ...     candidate_mask=candidate_mask,
    ...     extant_mask=extant_mask,
    ...     mark_as="is_persistent",
    ... )

    To then drop all but these leaves' lineages:

    >>> persistence_df = alifestd_prune_extinct_lineages_asexual(
    ...     persistence_df, criterion="is_persistent"
    ... )

    To instead mark by most recent common ancestor, whatever leaves' own
    branch lengths, make leaves contemporaneous first:

    >>> persistence_df = alifestd_mask_persistence_asexual(
    ...     alifestd_ultrametricize(df),
    ...     candidate_mask=candidate_mask,
    ...     extant_mask=extant_mask,
    ...     mark_as="is_persistent",
    ... )
    """
    if not mutate:
        phylogeny_df = phylogeny_df.copy()

    if criterion not in phylogeny_df.columns:
        raise ValueError(
            f"criterion column {criterion!r} not found in phylogeny_df",
        )

    phylogeny_df = alifestd_try_add_ancestor_id_col(phylogeny_df, mutate=True)
    extant_mask = np.asarray(extant_mask)
    candidate_mask = np.asarray(candidate_mask)
    for name, mask in (
        ("extant_mask", extant_mask),
        ("candidate_mask", candidate_mask),
    ):
        if mask.dtype != np.bool_:
            raise NotImplementedError(
                f"{name} must have boolean dtype, got {mask.dtype}",
            )
    if not alifestd_is_working_format_asexual(phylogeny_df, mutate=True):
        raise NotImplementedError(
            "current implementation requires phylogeny_df in working format",
        )

    ancestor_ids = phylogeny_df["ancestor_id"].to_numpy()
    criterion_values = phylogeny_df[criterion].to_numpy(dtype=np.float64)
    if not np.isfinite(criterion_values).all():
        raise NotImplementedError("non-finite criterion values not supported")
    if (criterion_values < criterion_values[ancestor_ids]).any():
        raise NotImplementedError(
            "criterion values decreasing from ancestor to descendant "
            "not supported",
        )

    phylogeny_df[mark_as] = _alifestd_mask_persistence_asexual_fast_path(
        ancestor_ids,
        extant_mask,
        candidate_mask,
        criterion_values,
    )
    return phylogeny_df
