import os

import pandas as pd
import pytest

from phyloframe.legacy import (
    alifestd_collapse_unifurcations,
    alifestd_mark_unifurcation_length_asexual,
    alifestd_to_working_format,
)

assets_path = os.path.join(os.path.dirname(__file__), "assets")


def _make_df() -> pd.DataFrame:
    """Tree:
    0 (root)
    +-- 1
        +-- 2
            +-- 3
            |   +-- 5
            |       +-- 6
            +-- 4
    """
    return pd.DataFrame(
        {
            "id": [0, 1, 2, 3, 4, 5, 6],
            "ancestor_id": [0, 0, 1, 2, 2, 3, 5],
            "w": [0.0, 4.0, 1.0, 0.0, 2.0, 0.0, 3.0],
        },
    )


@pytest.mark.parametrize("mutate", [True, False])
def test_simple_tree(mutate: bool):
    df = _make_df()
    original_df = df.copy()
    res = alifestd_mark_unifurcation_length_asexual(df, mutate=mutate)
    assert res["unifurcation_length"].tolist() == [0, 0, 1, 0, 0, 1, 2]
    if not mutate:
        assert df.equals(original_df)


def test_weights():
    res = alifestd_mark_unifurcation_length_asexual(_make_df(), weights="w")
    assert res["unifurcation_length"].tolist() == [0, 0, 4, 0, 0, 0, 0]


def test_mark_as():
    res = alifestd_mark_unifurcation_length_asexual(
        _make_df(), mark_as="blueberry"
    )
    assert "blueberry" in res.columns
    assert "unifurcation_length" not in res.columns


def test_unifurcating_root_not_counted():
    df = pd.DataFrame({"id": [0, 1, 2], "ancestor_id": [0, 0, 1]})
    res = alifestd_mark_unifurcation_length_asexual(df)
    assert res["unifurcation_length"].tolist() == [0, 0, 1]


def test_empty():
    df = pd.DataFrame({"id": [], "ancestor_id": []}, dtype=int)
    res = alifestd_mark_unifurcation_length_asexual(df)
    assert "unifurcation_length" in res.columns
    assert len(res) == 0


def test_missing_weights_column():
    with pytest.raises(ValueError):
        alifestd_mark_unifurcation_length_asexual(_make_df(), weights="nope")


def test_non_contiguous_ids():
    df = pd.DataFrame({"id": [0, 2], "ancestor_id": [0, 0]})
    with pytest.raises(NotImplementedError):
        alifestd_mark_unifurcation_length_asexual(df)


@pytest.mark.parametrize(
    "phylogeny_df",
    [
        pd.read_csv(
            f"{assets_path}/example-standard-toy-asexual-phylogeny.csv"
        ),
        pd.read_csv(f"{assets_path}/nk_tournamentselection.csv"),
    ],
)
def test_total_matches_collapse(phylogeny_df: pd.DataFrame):
    phylogeny_df = alifestd_to_working_format(phylogeny_df)
    res = alifestd_mark_unifurcation_length_asexual(phylogeny_df)
    collapsed = alifestd_collapse_unifurcations(res)
    assert collapsed["unifurcation_length"].sum() == len(phylogeny_df) - len(
        collapsed
    )
