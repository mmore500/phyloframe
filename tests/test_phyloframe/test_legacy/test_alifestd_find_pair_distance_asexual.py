import os

import pandas as pd
import pytest

from phyloframe.legacy import alifestd_to_working_format
from phyloframe.legacy._alifestd_find_pair_distance_asexual import (
    alifestd_find_pair_distance_asexual as alifestd_find_pair_distance_asexual_,
)

from ._impl import enforce_dtype_stability_pandas

alifestd_find_pair_distance_asexual = enforce_dtype_stability_pandas(
    alifestd_find_pair_distance_asexual_
)

assets_path = os.path.join(os.path.dirname(__file__), "assets")


@pytest.mark.parametrize("mutate", [True, False])
def test_simple1(mutate: bool):
    # Tree:  0 -> 1 -> 2, 0 -> 3
    # origin_times: 0=0, 1=10, 2=20, 3=15
    phylogeny_df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "ancestor_id": [0, 0, 1, 0],
            "origin_time": [0, 10, 20, 15],
        }
    )
    original_df = phylogeny_df.copy()

    # MRCA of 2 and 3 is 0; distance = (20-0) + (15-0) = 35
    assert alifestd_find_pair_distance_asexual(
        phylogeny_df, 2, 3, mutate=mutate
    ) == pytest.approx(35.0)
    # MRCA of 1 and 2 is 1; distance = (10-10) + (20-10) = 10
    assert alifestd_find_pair_distance_asexual(
        phylogeny_df, 1, 2, mutate=mutate
    ) == pytest.approx(10.0)
    # MRCA of 0 and 1 is 0; distance = (0-0) + (10-0) = 10
    assert alifestd_find_pair_distance_asexual(
        phylogeny_df, 0, 1, mutate=mutate
    ) == pytest.approx(10.0)
    # MRCA of a node with itself is itself; distance = 0
    assert alifestd_find_pair_distance_asexual(
        phylogeny_df, 2, 2, mutate=mutate
    ) == pytest.approx(0.0)
    assert alifestd_find_pair_distance_asexual(
        phylogeny_df, 0, 0, mutate=mutate
    ) == pytest.approx(0.0)

    if not mutate:
        assert original_df.equals(phylogeny_df)


def test_no_common_ancestor_returns_none():
    """Disjoint trees should return None."""
    phylogeny_df = pd.DataFrame(
        {
            "id": [0, 1, 2],
            "ancestor_id": [0, 1, 2],
            "origin_time": [0, 5, 10],
        }
    )
    assert alifestd_find_pair_distance_asexual(phylogeny_df, 0, 1) is None
    assert alifestd_find_pair_distance_asexual(phylogeny_df, 1, 2) is None


def test_single_node():
    phylogeny_df = pd.DataFrame(
        {
            "id": [0],
            "ancestor_id": [0],
            "origin_time": [42],
        }
    )
    assert alifestd_find_pair_distance_asexual(
        phylogeny_df, 0, 0
    ) == pytest.approx(0.0)


def test_custom_criterion():
    # Tree: 0 -> 1 -> 2, 0 -> 3
    phylogeny_df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "ancestor_id": [0, 0, 1, 0],
            "origin_time": [0, 10, 20, 15],
            "depth": [0.0, 1.0, 2.0, 1.0],
        }
    )
    # MRCA of 2 and 3 is 0; depth distance = (2.0-0.0) + (1.0-0.0) = 3.0
    assert alifestd_find_pair_distance_asexual(
        phylogeny_df, 2, 3, criterion="depth"
    ) == pytest.approx(3.0)


def test_chain():
    """Straight chain: 0 -> 1 -> 2 -> 3."""
    phylogeny_df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "ancestor_id": [0, 0, 1, 2],
            "origin_time": [0, 10, 20, 30],
        }
    )
    # MRCA of 0 and 3 is 0; distance = (0-0) + (30-0) = 30
    assert alifestd_find_pair_distance_asexual(
        phylogeny_df, 0, 3
    ) == pytest.approx(30.0)
    # MRCA of 1 and 3 is 1; distance = (10-10) + (30-10) = 20
    assert alifestd_find_pair_distance_asexual(
        phylogeny_df, 1, 3
    ) == pytest.approx(20.0)
    # MRCA of 2 and 3 is 2; distance = (20-20) + (30-20) = 10
    assert alifestd_find_pair_distance_asexual(
        phylogeny_df, 2, 3
    ) == pytest.approx(10.0)


def test_multiple_roots_partial():
    """Forest: tree1 = {0, 1}, tree2 = {2, 3}."""
    phylogeny_df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "ancestor_id": [0, 0, 2, 2],
            "origin_time": [0, 5, 10, 15],
        }
    )
    # Within tree1: MRCA of 0 and 1 is 0; distance = (0-0) + (5-0) = 5
    assert alifestd_find_pair_distance_asexual(
        phylogeny_df, 0, 1
    ) == pytest.approx(5.0)
    # Across trees: no common ancestor
    assert alifestd_find_pair_distance_asexual(phylogeny_df, 0, 2) is None


def test_simple_with_ancestor_list():
    phylogeny_df = alifestd_to_working_format(
        pd.DataFrame(
            {
                "id": [0, 1, 2, 3],
                "ancestor_list": ["[None]", "[0]", "[1]", "[0]"],
                "origin_time": [0, 10, 20, 15],
            }
        )
    )
    # MRCA of 2 and 3 is 0; distance = (20-0) + (15-0) = 35
    assert alifestd_find_pair_distance_asexual(
        phylogeny_df, 2, 3
    ) == pytest.approx(35.0)


@pytest.mark.parametrize(
    "phylogeny_df",
    [
        alifestd_to_working_format(
            pd.read_csv(f"{assets_path}/nk_ecoeaselection.csv")
        ),
        alifestd_to_working_format(
            pd.read_csv(f"{assets_path}/nk_lexicaseselection.csv")
        ),
        alifestd_to_working_format(
            pd.read_csv(f"{assets_path}/nk_tournamentselection.csv")
        ),
    ],
)
def test_fuzz_symmetry(phylogeny_df: pd.DataFrame):
    """Distance should be symmetric: d(a,b) == d(b,a)."""
    ids = phylogeny_df["id"].tolist()
    for i in ids[:5]:
        for j in ids[:5]:
            d_ij = alifestd_find_pair_distance_asexual(
                phylogeny_df, i, j, mutate=False
            )
            d_ji = alifestd_find_pair_distance_asexual(
                phylogeny_df, j, i, mutate=False
            )
            assert d_ij == pytest.approx(d_ji)


@pytest.mark.parametrize(
    "phylogeny_df",
    [
        alifestd_to_working_format(
            pd.read_csv(f"{assets_path}/nk_ecoeaselection.csv")
        ),
        alifestd_to_working_format(
            pd.read_csv(f"{assets_path}/nk_lexicaseselection.csv")
        ),
        alifestd_to_working_format(
            pd.read_csv(f"{assets_path}/nk_tournamentselection.csv")
        ),
    ],
)
def test_fuzz_self_distance_zero(phylogeny_df: pd.DataFrame):
    """Distance from a node to itself should be 0."""
    ids = phylogeny_df["id"].tolist()
    for i in ids[:5]:
        d = alifestd_find_pair_distance_asexual(
            phylogeny_df, i, i, mutate=False
        )
        assert d == pytest.approx(0.0)
