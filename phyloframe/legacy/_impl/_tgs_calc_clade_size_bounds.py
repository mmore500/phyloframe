import numpy as np


def tgs_calc_clade_size_bounds(
    *,
    max_clade_size: int,
    groups_per_doubling: int,
) -> np.ndarray:
    """Implementation detail for `alifestd_tgs_allmdc_estimate_asexual`.

    Here, tgs stands for trait gain/selection.

    Returns smallest clade sizes of log-scaled clade-size groups of
    multiple leaves, `groups_per_doubling` per doubling, evenly spaced
    within each doubling: with 2 per doubling, 2, 3, 4, 6, 8, 12, 16, ...
    Ends with the first bound above `max_clade_size`, so that group `j`
    spans sizes from `bounds[j]` to `bounds[j + 1] - 1`.
    """
    if groups_per_doubling < 1:
        raise ValueError(
            "clade_size_groups_per_doubling must be at least 1, got "
            f"{groups_per_doubling}",
        )
    bounds = [2]
    while bounds[-1] <= max_clade_size:
        # next bound past the last, evenly subdividing its doubling
        doubling = 2 ** int(np.log2(bounds[-1]))
        step = doubling / groups_per_doubling
        bounds.append(
            max(bounds[-1] + 1, int(np.ceil(bounds[-1] + step - 1e-9)))
        )
    return np.array(bounds, dtype=np.int64)
