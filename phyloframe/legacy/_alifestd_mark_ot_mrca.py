import numpy as np
import sortedcontainers as sc


def _calc_ot_mrca_contiguous(
    ids: np.ndarray,
    ancestor_ids: np.ndarray,
    origin_times: np.ndarray,
    is_leaf: np.ndarray,
) -> tuple:
    """Core MRCA-over-time calculation for contiguous-id phylogenies.

    Returns (ot_mrca_id, ot_mrca_time_of, ot_mrca_time_since) arrays.
    """
    n = len(ids)
    if n == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )

    ot_mrca_id = np.empty(n, dtype=np.int64)
    ot_mrca_time_of = np.empty(n, dtype=np.float64)
    ot_mrca_time_since = np.empty(n, dtype=np.float64)

    # group by negated origin_time (process most recent first)
    bwd_origin_time = -origin_times
    sort_order = np.argsort(bwd_origin_time, kind="stable")
    sorted_bwd = bwd_origin_time[sort_order]

    # find group boundaries
    group_breaks = np.concatenate(
        ([0], np.where(np.diff(sorted_bwd) != 0)[0] + 1, [n]),
    )

    # initialize running_mrca_id: leaf with latest origin_time,
    # break ties by largest index
    if is_leaf.any():
        leaf_indices = np.where(is_leaf)[0]
        leaf_times = origin_times[leaf_indices]
        max_time = leaf_times.max()
        candidates = leaf_indices[leaf_times == max_time]
        running_mrca_id = int(ids[candidates[-1]])
    else:
        running_mrca_id = int(ids[-1])

    # process each origin_time group
    for g in range(len(group_breaks) - 1):
        grp_start = group_breaks[g]
        grp_end = group_breaks[g + 1]
        grp_indices = sort_order[grp_start:grp_end]

        # collect active lineages
        grp_ids_leaf = ids[grp_indices[is_leaf[grp_indices]]]

        # earliest non-leaf in group (smallest index in contiguous case)
        grp_ids_all = ids[grp_indices]
        earliest_id = int(grp_ids_all.min())

        lineages = sc.SortedSet(
            [*grp_ids_leaf, earliest_id, running_mrca_id],
        )

        while len(lineages) > 1:
            oldest = lineages.pop(-1)
            replacement = int(ancestor_ids[oldest])
            assert replacement != oldest
            lineages.add(replacement)

        (mrca_id,) = lineages
        running_mrca_id = mrca_id

        mrca_time = float(origin_times[mrca_id])
        cur_origin_time = -sorted_bwd[grp_start]

        ot_mrca_id[grp_indices] = mrca_id
        ot_mrca_time_of[grp_indices] = mrca_time
        ot_mrca_time_since[grp_indices] = cur_origin_time - mrca_time

    return ot_mrca_id, ot_mrca_time_of, ot_mrca_time_since
