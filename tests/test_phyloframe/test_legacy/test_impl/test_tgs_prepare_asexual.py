import numpy as np
import pandas as pd
import pytest

from phyloframe.legacy._impl import tgs_prepare_asexual


def _make_tree() -> pd.DataFrame:
    #     0
    #   1   2
    #   3
    # node 1 is a unifurcation
    return pd.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "ancestor_id": [0, 0, 0, 1],
            "origin_time": [0.0, 1.0, 2.0, 5.0],
        },
    )


@pytest.mark.parametrize("mutate", [False, True])
def test_prepare(mutate: bool):
    phylogeny_df = _make_tree()
    original_df = phylogeny_df.copy()
    trait = np.array([False, True, False, True])
    res = tgs_prepare_asexual(
        phylogeny_df, mutate, trait_masks=[trait], trait_names=["t"]
    )
    if not mutate:
        assert original_df.equals(phylogeny_df)
    # unifurcation 1 is bypassed, so leaf 3 hangs from root 0
    assert len(res) == 3
    by_time = res.set_index("origin_time")
    assert by_time.loc[5.0, "__alifestd_tgs_branch_length"] == 5.0
    assert by_time.loc[2.0, "__alifestd_tgs_branch_length"] == 2.0
    assert by_time.loc[5.0, "__alifestd_tgs_trait0"]
    assert not by_time.loc[2.0, "__alifestd_tgs_trait0"]


def test_mask_validated():
    with pytest.raises(NotImplementedError):
        tgs_prepare_asexual(
            _make_tree(),
            False,
            trait_masks=[np.array([0, 1, 0, 1])],
            trait_names=["t"],
        )
    with pytest.raises(ValueError):
        tgs_prepare_asexual(
            _make_tree(),
            False,
            trait_masks=[np.zeros(3, dtype=bool)],
            trait_names=["t"],
        )
