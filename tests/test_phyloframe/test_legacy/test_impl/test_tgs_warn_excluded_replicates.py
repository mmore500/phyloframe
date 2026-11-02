import warnings

import pandas as pd
import pytest

from phyloframe.legacy._impl import tgs_warn_excluded_replicates


def _excluded(values) -> pd.Series:
    return pd.Series(
        values,
        index=pd.MultiIndex.from_product(
            [range(2), range(2)], names=["replicate", "trait_index"]
        ),
    )


def test_warns_with_counts():
    with pytest.warns(UserWarning, match=r"traits b \(2 of 2\)"):
        tgs_warn_excluded_replicates(
            _excluded([False, True, False, True]),
            trait_names=["a", "b"],
            num_replicates=2,
            reason="no carriers of the trait",
        )


def test_no_warning_without_exclusions():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        tgs_warn_excluded_replicates(
            _excluded([False] * 4),
            trait_names=["a", "b"],
            num_replicates=2,
            reason="no carriers of the trait",
        )
