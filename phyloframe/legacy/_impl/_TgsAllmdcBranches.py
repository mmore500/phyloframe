import typing

import numpy as np


class TgsAllmdcBranches(typing.NamedTuple):
    """Implementation detail for `alifestd_tgs_allmdc_estimate_asexual`.

    Here, tgs stands for trait gain/selection.

    Branches informative for fitting (positive length), with the number of
    leaves each subtends (1 for a leaf); each branch's length; its survival time, from
    its end node to the latest sampled leaf below it (0 for a leaf); its
    end node's origin time; and per trait, positions of branches that
    started without the trait and of those that gained it.
    """

    clade_sizes: np.ndarray
    branch_lengths: np.ndarray
    survival_times: np.ndarray
    origin_times: np.ndarray
    trait_units: typing.List[np.ndarray]
    trait_gains: typing.List[np.ndarray]
