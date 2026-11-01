import typing

import numpy as np


class TgsCounts(typing.NamedTuple):
    """Implementation detail for
    `alifestd_tgs_pendantmdc_estimate_asexual`.

    Here, tgs stands for trait gain/selection.

    Per-trait (weighted) counts in each partition, as arrays of shape
    (num_traits, num_partitions).
    """

    num_leaves: np.ndarray
    num_carriers: np.ndarray
    pendant_length_sums: np.ndarray
