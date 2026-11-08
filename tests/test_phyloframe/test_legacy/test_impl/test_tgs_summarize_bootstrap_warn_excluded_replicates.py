import warnings

import numpy as np
import pytest

from phyloframe.legacy._impl._tgs_summarize_bootstrap import (
    _warn_excluded_replicates,
)

# two replicates of two traits
_TRAIT_IDS = np.array([0, 1, 0, 1])


def test_warns_with_counts():
    with pytest.warns(UserWarning, match=r"traits b \(2 of 2\)"):
        _warn_excluded_replicates(
            np.array([False, True, False, True]),
            trait_ids=_TRAIT_IDS,
            trait_names=["a", "b"],
            num_replicates=2,
            reason="no carriers of the trait",
        )


def test_no_warning_without_exclusions():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _warn_excluded_replicates(
            np.zeros(4, dtype=bool),
            trait_ids=_TRAIT_IDS,
            trait_names=["a", "b"],
            num_replicates=2,
            reason="no carriers of the trait",
        )
