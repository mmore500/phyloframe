import typing

import numpy as np
from scipy import sparse


class TgsAllmdcGroup(typing.NamedTuple):
    """Implementation detail for `alifestd_tgs_allmdc_estimate_asexual`.

    Here, tgs stands for trait gain/selection.

    Branches subtending clades of one size group, binned into partitions:
    each branch's partition and length, per trait the positions (within the
    group) of branches that started without the trait and of those that
    gained it, and partition-by-partition calibration neighborhoods (None
    without calibration).
    """

    partitions: np.ndarray
    num_partitions: int
    branch_lengths: np.ndarray
    trait_units: typing.List[np.ndarray]
    trait_gains: typing.List[np.ndarray]
    calibration_neighborhoods: typing.Optional[sparse.spmatrix]
