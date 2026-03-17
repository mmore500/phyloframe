import typing

import numpy as np

from ._jit import jit


@jit(nopython=True)
def build_children_csr(
    ancestor_ids: np.ndarray,
    num_children: np.ndarray,
) -> typing.Tuple[np.ndarray, np.ndarray]:
    """Build CSR-style children arrays from ancestor ids and child counts.

    Parameters
    ----------
    ancestor_ids : np.ndarray
        Array of ancestor IDs, assumed contiguous (ids == row indices)
        and topologically sorted.
    num_children : np.ndarray
        Array of child counts per node.

    Returns
    -------
    child_start : np.ndarray
        CSR offset array of length n+1.
    csr_children : np.ndarray
        Flat array of child ids, grouped by parent.
    """
    n = len(ancestor_ids)

    child_start = np.zeros(n + 1, dtype=np.int64)
    for i, cc in enumerate(num_children):
        child_start[i + 1] = child_start[i] + cc

    csr_children = np.empty(n, dtype=np.int64)
    insert_pos = child_start[:-1].copy()
    for i, p in enumerate(ancestor_ids):
        if p != i:
            csr_children[insert_pos[p]] = i
            insert_pos[p] += 1

    return child_start, csr_children
