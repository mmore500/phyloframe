import numpy as np


def calc_group_means(
    groups: np.ndarray, values: np.ndarray, *, num_groups: int
) -> np.ndarray:
    """Calculate the mean of `values` within each group.

    Parameters
    ----------
    groups : np.ndarray
        Integer group of each item, in `range(num_groups)`.
    values : np.ndarray
        Array of shape (num_items,) or (num_items, num_columns).
    num_groups : int
        Number of groups.

    Returns
    -------
    np.ndarray
        Array of shape (num_groups,) or (num_groups, num_columns), with
        each group's mean; nan for groups without items.
    """
    values = np.asarray(values, dtype=np.float64)
    columns = values.T if values.ndim > 1 else values[None, :]
    sizes = np.bincount(groups, minlength=num_groups)
    means = np.column_stack(
        [
            np.divide(
                np.bincount(groups, column, minlength=num_groups),
                sizes,
                out=np.full(num_groups, np.nan),
                where=sizes > 0,
            )
            for column in columns
        ],
    )
    return means.reshape((num_groups,) + values.shape[1:])
