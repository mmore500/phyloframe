import numpy as np
import polars as pl


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
    groups = np.asarray(groups)
    values = np.asarray(values, dtype=np.float64)
    assert groups.ndim == 1
    assert values.ndim in (1, 2)
    assert len(groups) == len(values)
    assert (np.clip(groups, 0, num_groups - 1) == groups).all()
    df = pl.DataFrame(np.column_stack([values]), orient="row").with_columns(
        group=pl.Series(groups, dtype=pl.Int64)
    )
    # an empty row per group, so that groups without items get null (nan)
    # means rather than being dropped
    placeholders = pl.DataFrame({"group": np.arange(num_groups)})
    means = (
        pl.concat([df, placeholders], how="diagonal")
        .group_by("group")
        .mean()
        .sort("group")
    )
    return (
        means.drop("group").to_numpy().reshape((num_groups, *values.shape[1:]))
    )
