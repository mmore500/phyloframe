import numpy as np
import polars as pl
import pytest

from phyloframe.legacy._impl._tgs_build_exposure_df_polars import (
    _tgs_build_exposure_df,
)


def _exposure_df(*, ancestor_ids, origin_time, trait_states):
    return _tgs_build_exposure_df(
        pl.DataFrame(
            {
                "ancestor_id": ancestor_ids,
                "origin_time": origin_time,
                **{
                    f"trait{i}_state": states
                    for i, states in enumerate(trait_states)
                },
            },
        ),
        episode_mask=np.ones(len(ancestor_ids), dtype=bool),
    )


def test_branches():
    #         0
    #      1     6
    #    2   5
    #   3 4
    ancestor_ids = np.array([0, 0, 1, 2, 2, 1, 0])
    origin_time = np.array([0.0, 1.0, 2.0, 4.0, 5.0, 3.0, 2.0])
    trait_states = [
        np.array([0, 0, 1, 1, 1, 0, 0]),  # gained on the branch into 2
        np.array([1, 1, 1, 1, 1, 1, 1]),  # present from the root
    ]
    res = _exposure_df(
        ancestor_ids=ancestor_ids,
        origin_time=origin_time,
        trait_states=trait_states,
    )
    # the root has no branch
    assert res["origin_time"].to_list() == [1.0, 2.0, 4.0, 5.0, 3.0, 2.0]
    assert res["branch_length"].to_list() == [1.0, 1.0, 2.0, 3.0, 2.0, 2.0]
    # to the latest leaf below: node 4 (time 5) for nodes 1 and 2
    assert res["survival_duration"].to_list() == [4.0, 3.0, 0.0, 0.0, 0.0, 0.0]
    assert res["clade_size"].to_list() == [3, 2, 1, 1, 1, 1]
    assert res["trait0_starts_absent"].to_list() == [1, 1, 0, 0, 1, 1]
    assert res["trait0_gain"].to_list() == [0, 1, 0, 0, 0, 0]
    assert not res["trait1_starts_absent"].any()
    assert not res["trait1_gain"].any()


@pytest.mark.parametrize("bad_origin_time", [0.0, -1.0, np.nan])
def test_nonpositive_length_raises(bad_origin_time: float):
    with pytest.raises(ValueError):
        _exposure_df(
            ancestor_ids=np.array([0, 0, 0]),
            origin_time=np.array([0.0, bad_origin_time, 1.0]),
            trait_states=[np.array([0, 0, 1])],
        )


def test_root_length_ignored():
    res = _exposure_df(
        ancestor_ids=np.array([0, 0, 2]),
        origin_time=np.array([0.0, 1.0, np.nan]),
        trait_states=[np.array([0, 1, 0])],
    )
    assert res["branch_length"].to_list() == [1.0]
    assert res["trait0_gain"].to_list() == [True]
