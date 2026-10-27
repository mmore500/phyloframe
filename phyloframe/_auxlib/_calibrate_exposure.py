import numpy as np


def calibrate_exposure(
    num_successes: np.ndarray,
    exposure_total: np.ndarray,
    cell_ids: np.ndarray,
) -> np.ndarray:
    """Learn a per-cell exposure correction from reference counts known to
    accrue at a constant rate.

    Within each cell, effective exposure is taken as the reference success
    count divided by the overall reference rate, so the returned factor is
    the cell's reference rate relative to the overall reference rate.
    Multiplying measured exposure by this factor removes distortion of
    measured exposure that the reference shares.

    Parameters
    ----------
    num_successes : np.ndarray
        Reference success count in each bin.
    exposure_total : np.ndarray
        Summed reference exposure in each bin.
    cell_ids : np.ndarray
        Calibration cell of each bin, broadcastable to the shape of
        `num_successes`; bins in the same cell share a correction factor.

    Returns
    -------
    np.ndarray
        Multiplicative exposure correction for each bin, with the shape of
        `num_successes`. Bins in cells with no reference exposure get 1.0.
    """
    num_successes = np.asarray(num_successes, dtype=np.float64)
    exposure_total = np.asarray(exposure_total, dtype=np.float64)
    cell_ids = np.broadcast_to(cell_ids, num_successes.shape).ravel()
    _, cell_ids = np.unique(cell_ids, return_inverse=True)

    cell_successes = np.bincount(cell_ids, weights=num_successes.ravel())
    cell_exposure = np.bincount(cell_ids, weights=exposure_total.ravel())
    overall_rate = cell_successes.sum() / max(cell_exposure.sum(), 1e-300)
    if overall_rate == 0:
        return np.ones_like(num_successes)

    factor = np.where(
        cell_exposure > 0,
        cell_successes / np.maximum(cell_exposure, 1e-300) / overall_rate,
        1.0,
    )
    return factor[cell_ids].reshape(num_successes.shape)
