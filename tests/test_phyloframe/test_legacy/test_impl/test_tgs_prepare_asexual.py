import pandas as pd
import pytest

from phyloframe.legacy._impl import tgs_prepare_asexual


def _make_tree() -> pd.DataFrame:
    #     0
    #   1   2
    #   3
    # node 1 is a unifurcation; trait t is gained on the branch into 1
    return pd.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "ancestor_id": [0, 0, 0, 1],
            "origin_time": [0.0, 1.0, 2.0, 5.0],
            "t": [0, 1, 0, 0],
        },
    )


@pytest.mark.parametrize("mutate", [False, True])
def test_prepare(mutate: bool):
    phylogeny_df = _make_tree()
    original_df = phylogeny_df.copy()
    res = tgs_prepare_asexual(
        phylogeny_df, mutate, trait_transitions=["t"], trait_names=["t"]
    )
    if not mutate:
        pd.testing.assert_frame_equal(phylogeny_df, original_df)
    # unifurcation 1 is bypassed, so leaf 3 hangs from root 0, and the
    # gain into 1 is credited to the merged branch
    by_time = res.set_index("origin_time")
    assert sorted(by_time.index) == [2.0, 5.0]
    assert by_time.loc[5.0, "branch_length"] == 5.0
    assert by_time.loc[2.0, "branch_length"] == 2.0
    assert by_time["clade_size"].tolist() == [1, 1]
    assert by_time["survival_time"].tolist() == [0.0, 0.0]
    assert by_time["trait0_at_risk"].all()
    assert by_time.loc[5.0, "trait0_gain"]
    assert not by_time.loc[2.0, "trait0_gain"]


def test_missing_column_raises():
    with pytest.raises(ValueError):
        tgs_prepare_asexual(
            _make_tree(), False, trait_transitions=["x"], trait_names=["x"]
        )


def test_unsorted_input():
    phylogeny_df = _make_tree().iloc[::-1].reset_index(drop=True)
    res = tgs_prepare_asexual(
        phylogeny_df, False, trait_transitions=["t"], trait_names=["t"]
    )
    expected = tgs_prepare_asexual(
        _make_tree(), False, trait_transitions=["t"], trait_names=["t"]
    )
    pd.testing.assert_frame_equal(
        res.sort_values("origin_time").reset_index(drop=True),
        expected.sort_values("origin_time").reset_index(drop=True),
    )
