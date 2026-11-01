import typing

import numpy as np


class TgsAllmdcUnits(typing.NamedTuple):
    """Implementation detail for `alifestd_tgs_allmdc_estimate_asexual`.

    Here, tgs stands for trait gain/selection.

    Informative branches by clade-size group: the nodes subtending each
    group's branches, each group's (fewest, most) leaves, and per group and
    trait, the positions (within the group) of branches that started
    without the trait and of those that gained it.
    """

    group_nodes: typing.List[np.ndarray]
    group_bounds: typing.List[typing.Tuple[int, int]]
    trait_units: typing.List[typing.List[np.ndarray]]
    trait_gains: typing.List[typing.List[np.ndarray]]
