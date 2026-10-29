import numpy as np
from scipy import stats


def calc_two_sided_normal_p(z: np.ndarray) -> np.ndarray:
    """Calculate the two-sided p-value of each z-score under a standard
    normal distribution.

    Parameters
    ----------
    z : np.ndarray
        Z-scores; nan gives nan.

    Returns
    -------
    np.ndarray
        Probability of a standard normal deviate at least as far from zero
        as each z-score, in either direction.
    """
    return 2 * stats.norm.sf(np.abs(z))
