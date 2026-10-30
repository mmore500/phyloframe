import os

import numpy as np
import pandas as pd
import pytest

from phyloframe.legacy import (
    alifestd_assign_contiguous_ids,
    alifestd_collapse_unifurcations,
    alifestd_collapse_unifurcations_tallied_asexual,
    alifestd_to_working_format,
)

assets_path = os.path.join(os.path.dirname(__file__), "assets")


def _make_df() -> pd.DataFrame:
    """Tree:
    0 (root)
    +-- 1 (unifurcation)
        +-- 2
            +-- 3 (unifurcation)
            |   +-- 5 (unifurcation)
            |       +-- 6
            +-- 4
    """
    return pd.DataFrame(
        {
            "id": [0, 1, 2, 3, 4, 5, 6],
            "ancestor_id": [0, 0, 1, 2, 2, 3, 5],
            "tally": np.array([0, 4, 1, 0, 2, 0, 3], dtype=np.uint32),
        },
    )


@pytest.mark.parametrize("mutate", [True, False])
def test_simple_tree(mutate: bool):
    df = _make_df()
    original_df = df.copy()
    res = alifestd_collapse_unifurcations_tallied_asexual(
        df, mutate=mutate, tally_as="tally"
    )
    tallies = dict(zip(res["id"], res["tally"]))
    # node 2 tallies itself, plus node 1 and node 1's prior tally of 4
    # node 6 tallies itself, plus nodes 3 and 5 (prior tallies 0)
    assert tallies == {0: 0, 2: 1 + 1 + 4, 4: 2, 6: 3 + 2}
    assert res["tally"].dtype == np.uint32
    if not mutate:
        pd.testing.assert_frame_equal(df, original_df)


def test_matches_untallied_topology():
    df = _make_df()
    res = alifestd_collapse_unifurcations_tallied_asexual(df, tally_as="tally")
    expected = alifestd_collapse_unifurcations(df)
    pd.testing.assert_frame_equal(
        res.drop(columns="tally").reset_index(drop=True),
        expected.drop(columns="tally").reset_index(drop=True),
    )


def test_missing_tally_starts_from_zero():
    df = _make_df().drop(columns="tally")
    res = alifestd_collapse_unifurcations_tallied_asexual(df)
    tallies = dict(zip(res["id"], res["collapsed_unifurcations"]))
    assert tallies == {0: 0, 2: 1, 4: 0, 6: 2}


@pytest.mark.parametrize(
    "phylogeny_df",
    [
        pd.read_csv(f"{assets_path}/nk_ecoeaselection.csv"),
        pd.read_csv(f"{assets_path}/collapse_unifurcations_testphylo.csv"),
    ],
)
def test_tally_conserves_node_count(phylogeny_df: pd.DataFrame):
    df = alifestd_to_working_format(phylogeny_df)
    res = alifestd_collapse_unifurcations_tallied_asexual(df)
    assert res["collapsed_unifurcations"].sum() == len(df) - len(res)


def test_repeated_collapse_is_stable():
    once = alifestd_collapse_unifurcations_tallied_asexual(
        _make_df(), tally_as="tally"
    )
    twice = alifestd_collapse_unifurcations_tallied_asexual(
        alifestd_assign_contiguous_ids(once), tally_as="tally"
    )
    # no unifurcations remain, so tallies are unchanged
    assert twice["tally"].tolist() == once["tally"].tolist()


def test_root_unifurcation_kept():
    df = pd.DataFrame({"id": [0, 1], "ancestor_id": [0, 0]})
    res = alifestd_collapse_unifurcations_tallied_asexual(df)
    assert res["id"].tolist() == [0, 1]
    assert res["collapsed_unifurcations"].tolist() == [0, 0]


def test_empty():
    df = pd.DataFrame({"id": [], "ancestor_id": []}, dtype=int)
    res = alifestd_collapse_unifurcations_tallied_asexual(df)
    assert "collapsed_unifurcations" in res.columns
    assert len(res) == 0


def test_nan_tally_not_implemented():
    df = _make_df()
    df["tally"] = df["tally"].astype(float)
    df.loc[2, "tally"] = np.nan
    with pytest.raises(NotImplementedError):
        alifestd_collapse_unifurcations_tallied_asexual(df, tally_as="tally")


def test_non_contiguous_ids_not_implemented():
    df = pd.DataFrame({"id": [0, 2], "ancestor_id": [0, 0]})
    with pytest.raises(NotImplementedError):
        alifestd_collapse_unifurcations_tallied_asexual(df)
