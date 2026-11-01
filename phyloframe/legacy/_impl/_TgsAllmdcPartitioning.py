import typing

import numpy as np
from scipy import sparse


class TgsAllmdcPartitioning(typing.NamedTuple):
    """Implementation detail for `alifestd_tgs_allmdc_estimate_asexual`.

    Here, tgs stands for trait gain/selection.

    Each branch's partition (by clade-size block, then ranked length and
    origin time) and survival-time bin; each bin's mean survival time; and
    partition-by-partition calibration neighborhoods (None without
    calibration).
    """

    partitions: np.ndarray
    num_partitions: int
    survival_bins: np.ndarray
    survival_bin_times: np.ndarray
    calibration_neighborhoods: typing.Optional[sparse.spmatrix]
